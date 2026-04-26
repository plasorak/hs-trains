use crate::core::model::{Infrastructure, NetElement, OperationalPoint, Track};
use crate::io::xml_util::parse_attr;
use std::collections::HashMap;
use std::path::Path;

const NS: &str = "https://www.railml.org/schemas/3.3";
const GML_NS: &str = "http://www.opengis.net/gml/3.2";

/// Parse the `<infrastructure>` section of a RailML 3.3 file.
///
/// Extracts:
/// - `netElement` elements from `topology/netElements` (id, length in metres)
/// - `track` elements from `functionalInfrastructure/tracks` (id, netElementRef)
/// - `operationalPoint` elements from `functionalInfrastructure/operationalPoints` (id, name)
///
/// NetElements without a `@length` attribute are skipped with a warning.
pub fn load_infrastructure(path: &Path) -> Result<Infrastructure, String> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
    parse_infrastructure_xml(&xml, &path.display().to_string())
}

fn parse_infrastructure_xml(xml: &str, label: &str) -> Result<Infrastructure, String> {
    let doc = roxmltree::Document::parse(xml)
        .map_err(|e| format!("XML parse error in '{label}': {e}"))?;

    let infra_node = doc
        .descendants()
        .find(|n| n.has_tag_name((NS, "infrastructure")))
        .ok_or_else(|| format!("no <infrastructure> element in '{label}'"))?;

    // --- NetElements -----------------------------------------------------------

    let mut net_elements = HashMap::new();
    for ne in infra_node.descendants().filter(|n| n.has_tag_name((NS, "netElement"))) {
        let ctx = format!("netElement in '{label}'");
        let id: String = parse_attr(ne, "id", &ctx)?;
        // Accept both the hand-authored shorthand (@length="x") and the pydantic-generated
        // RailML 3.3 form (<rail3:length quantity="x"/>).
        let length_m: f64 = if let Ok(v) = parse_attr::<f64>(ne, "length", &ctx) {
            v
        } else if let Some(child) = ne.children().find(|n| n.has_tag_name((NS, "length"))) {
            match child.attribute("quantity").and_then(|v| v.parse::<f64>().ok()) {
                Some(v) => v,
                None => {
                    eprintln!("Warning: netElement '{id}' has no parseable length — skipped");
                    continue;
                }
            }
        } else {
            eprintln!("Warning: netElement '{id}' has no parseable length — skipped");
            continue;
        };
        net_elements.insert(id.clone(), NetElement { id, length_m });
    }

    // --- Tracks ----------------------------------------------------------------

    let mut tracks = HashMap::new();
    let mut track_coords: HashMap<String, Vec<(f64, f64)>> = HashMap::new();
    for track in infra_node.descendants().filter(|n| n.has_tag_name((NS, "track"))) {
        let ctx = format!("track in '{label}'");
        let id: String = parse_attr(track, "id", &ctx)?;
        let net_element_id: String = parse_attr(track, "netElementRef", &ctx)?;

        // Parse optional GML geometry — space-separated "lon lat" pairs in posList.
        let coords: Vec<(f64, f64)> = track
            .descendants()
            .find(|n| n.has_tag_name((GML_NS, "posList")))
            .and_then(|n| n.text())
            .map(|text| {
                let nums: Vec<f64> =
                    text.split_whitespace().filter_map(|s| s.parse().ok()).collect();
                nums.chunks(2)
                    .filter_map(|c| if c.len() == 2 { Some((c[0], c[1])) } else { None })
                    .collect()
            })
            .unwrap_or_default();
        track_coords.insert(id.clone(), coords);
        tracks.insert(id.clone(), Track { id, net_element_id });
    }

    // --- OperationalPoints (informational only) ---------------------------------

    let mut ops = HashMap::new();
    for op in infra_node.descendants().filter(|n| n.has_tag_name((NS, "operationalPoint"))) {
        if let Some(id) = op.attribute("id") {
            let name = op.attribute("name").map(str::to_string);
            ops.insert(id.to_string(), OperationalPoint { id: id.to_string(), name });
        }
    }

    Ok(Infrastructure { net_elements, tracks, ops, track_coords })
}

#[cfg(test)]
mod tests {
    use super::*;

    const NS_DECL: &str = r#"xmlns:rail3="https://www.railml.org/schemas/3.3""#;

    fn xml(body: &str) -> String {
        format!(r#"<?xml version="1.0"?><rail3:railml {NS_DECL}>{body}</rail3:railml>"#)
    }

    fn infra(body: &str) -> String {
        xml(&format!("<rail3:infrastructure>{body}</rail3:infrastructure>"))
    }

    #[test]
    fn test_basic_net_elements_and_tracks() {
        let doc = infra(
            r#"<rail3:topology>
                 <rail3:netElements>
                   <rail3:netElement id="ne_1" length="1000.0"/>
                   <rail3:netElement id="ne_2" length="500.5"/>
                 </rail3:netElements>
               </rail3:topology>
               <rail3:functionalInfrastructure>
                 <rail3:tracks>
                   <rail3:track id="track_A" netElementRef="ne_1"/>
                   <rail3:track id="track_B" netElementRef="ne_2"/>
                 </rail3:tracks>
               </rail3:functionalInfrastructure>"#,
        );
        let infra = parse_infrastructure_xml(&doc, "test").unwrap();

        assert_eq!(infra.net_elements.len(), 2);
        assert!((infra.net_elements["ne_1"].length_m - 1000.0).abs() < 1e-9);
        assert!((infra.net_elements["ne_2"].length_m - 500.5).abs() < 1e-9);

        assert_eq!(infra.tracks.len(), 2);
        assert_eq!(infra.tracks["track_A"].net_element_id, "ne_1");
        assert_eq!(infra.tracks["track_B"].net_element_id, "ne_2");
    }

    #[test]
    fn test_operational_points_with_and_without_name() {
        let doc = infra(
            r#"<rail3:functionalInfrastructure>
                 <rail3:operationalPoints>
                   <rail3:operationalPoint id="OP_A" name="Station Alpha"/>
                   <rail3:operationalPoint id="OP_B"/>
                 </rail3:operationalPoints>
               </rail3:functionalInfrastructure>"#,
        );
        let infra = parse_infrastructure_xml(&doc, "test").unwrap();

        assert_eq!(infra.ops.len(), 2);
        assert_eq!(infra.ops["OP_A"].name.as_deref(), Some("Station Alpha"));
        assert!(infra.ops["OP_B"].name.is_none());
    }

    #[test]
    fn test_net_element_length_child_element_form() {
        // pydantic-generated RailML uses <rail3:length quantity="x"/> rather than @length="x"
        let doc = infra(
            r#"<rail3:topology>
                 <rail3:netElements>
                   <rail3:netElement id="ne_pydantic">
                     <rail3:length quantity="750.25"/>
                   </rail3:netElement>
                 </rail3:netElements>
               </rail3:topology>"#,
        );
        let infra = parse_infrastructure_xml(&doc, "test").unwrap();
        assert_eq!(infra.net_elements.len(), 1);
        assert!((infra.net_elements["ne_pydantic"].length_m - 750.25).abs() < 1e-9);
    }

    #[test]
    fn test_net_element_without_length_is_skipped() {
        let doc = infra(
            r#"<rail3:topology>
                 <rail3:netElements>
                   <rail3:netElement id="ne_good" length="200.0"/>
                   <rail3:netElement id="ne_bad"/>
                 </rail3:netElements>
               </rail3:topology>"#,
        );
        let infra = parse_infrastructure_xml(&doc, "test").unwrap();
        // ne_bad has no @length and must be silently dropped.
        assert_eq!(infra.net_elements.len(), 1);
        assert!(infra.net_elements.contains_key("ne_good"));
    }

    #[test]
    fn test_missing_infrastructure_element_returns_error() {
        let doc = xml("<rail3:rollingstock/>");
        let result = parse_infrastructure_xml(&doc, "test");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no <infrastructure>"));
    }

    #[test]
    fn test_empty_infrastructure_is_valid() {
        let doc = infra("");
        let infra = parse_infrastructure_xml(&doc, "test").unwrap();
        assert!(infra.net_elements.is_empty());
        assert!(infra.tracks.is_empty());
        assert!(infra.ops.is_empty());
    }

    #[test]
    fn test_gml_coords_parsed() {
        let doc = infra(
            r#"<rail3:functionalInfrastructure>
                 <rail3:tracks>
                   <rail3:track id="track_geo" netElementRef="ne_1">
                     <rail3:gmlLocation xmlns:gml="http://www.opengis.net/gml/3.2">
                       <gml:LineString srsName="urn:ogc:def:crs:EPSG::4326">
                         <gml:posList>-1.0 53.0 -1.1 53.1 -1.2 53.2</gml:posList>
                       </gml:LineString>
                     </rail3:gmlLocation>
                   </rail3:track>
                   <rail3:track id="track_no_geo" netElementRef="ne_2"/>
                 </rail3:tracks>
               </rail3:functionalInfrastructure>"#,
        );
        let infra = parse_infrastructure_xml(&doc, "test").unwrap();
        let coords = &infra.track_coords["track_geo"];
        assert_eq!(coords.len(), 3);
        assert!((coords[0].0 - (-1.0)).abs() < 1e-9); // lon
        assert!((coords[0].1 - 53.0).abs() < 1e-9);   // lat
        assert!((coords[2].0 - (-1.2)).abs() < 1e-9);
        // Track without GML gets empty vec.
        assert!(infra.track_coords["track_no_geo"].is_empty());
    }

    #[test]
    fn test_track_missing_net_element_ref_returns_error() {
        let doc = infra(
            r#"<rail3:functionalInfrastructure>
                 <rail3:tracks>
                   <rail3:track id="track_bad"/>
                 </rail3:tracks>
               </rail3:functionalInfrastructure>"#,
        );
        let result = parse_infrastructure_xml(&doc, "test");
        assert!(result.is_err());
    }
}
