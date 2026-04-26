use crate::core::model::{Infrastructure, Route, RouteElement};
use std::collections::HashMap;
use std::path::Path;

const NS: &str = "https://www.railml.org/schemas/3.3";

/// Parse routes for all operational trains from the `<timetable>` section of a
/// RailML 3.3 file.
///
/// Only the **route geometry** is extracted — no timing, stops, or validity.
///
/// ## Resolution chain
///
/// ```text
/// operationalTrain
///   operationalTrainVariant[@itineraryRef]
///     itinerary
///       range[@baseItineraryRef, @sequenceNumber, @start?, @end?]
///         baseItinerary
///           baseItineraryPoint[@sequenceNumber]  (sorted ascending)
///             followupSections/followupSection   (lowest priority value wins)
///               trackRefs/trackRef[@ref, @sequenceNumber]
/// ```
///
/// Track IDs are resolved to `RouteElement`s via the supplied `Infrastructure`.
/// Tracks or net elements that are absent from the infrastructure are skipped
/// with a warning — they do not abort the parse.
///
/// Returns a map from **operational train id** → `Route`.
pub fn load_routes(
    path: &Path,
    infra: &Infrastructure,
) -> Result<HashMap<String, Route>, String> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
    parse_routes_xml(&xml, &path.display().to_string(), infra)
}

fn parse_routes_xml(
    xml: &str,
    label: &str,
    infra: &Infrastructure,
) -> Result<HashMap<String, Route>, String> {
    let doc = roxmltree::Document::parse(xml)
        .map_err(|e| format!("XML parse error in '{label}': {e}"))?;

    let tt_node = match doc.descendants().find(|n| n.has_tag_name((NS, "timetable"))) {
        Some(n) => n,
        None => {
            eprintln!("No <timetable> element in '{label}' — no routes loaded");
            return Ok(HashMap::new());
        }
    };

    // -----------------------------------------------------------------------
    // Step 1: Parse every baseItinerary into an ordered list of track IDs.
    //
    // A baseItinerary is a sequence of baseItineraryPoints.  The tracks that
    // connect point N to point N+1 live in point N's followupSections.
    // The last point has no followupSections (no onward section).
    // -----------------------------------------------------------------------

    // bip_id → sequence_number (for range slicing later)
    let mut bip_seq: HashMap<String, u32> = HashMap::new();
    // base_itinerary_id → (bip_id, seq, Vec<(seq, track_id)>)
    let mut base_itineraries: HashMap<String, Vec<(u32, String, Vec<(u32, String)>)>> =
        HashMap::new();

    for bi in tt_node.descendants().filter(|n| n.has_tag_name((NS, "baseItinerary"))) {
        let bi_id = match bi.attribute("id") {
            Some(v) => v.to_string(),
            None => continue,
        };

        let mut points: Vec<(u32, String, Vec<(u32, String)>)> = Vec::new();

        for bip in bi.children().filter(|n| n.has_tag_name((NS, "baseItineraryPoint"))) {
            let bip_id = match bip.attribute("id") {
                Some(v) => v.to_string(),
                None => continue,
            };
            let seq: u32 = bip
                .attribute("sequenceNumber")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);

            bip_seq.insert(bip_id.clone(), seq);

            // Collect tracks from the best-priority followupSection.
            // Priority 0 (or absent) = highest priority; pick lowest numeric value.
            let followup_sections = bip
                .children()
                .find(|n| n.has_tag_name((NS, "followupSections")));

            let track_ids = match followup_sections {
                None => Vec::new(),
                Some(fs_container) => {
                    let best_section = fs_container
                        .children()
                        .filter(|n| n.has_tag_name((NS, "followupSection")))
                        .min_by_key(|n| {
                            n.attribute("priority")
                                .and_then(|v| v.parse::<i64>().ok())
                                .unwrap_or(0)
                        });

                    match best_section {
                        None => Vec::new(),
                        Some(section) => {
                            let track_refs_container = section
                                .children()
                                .find(|n| n.has_tag_name((NS, "trackRefs")));
                            match track_refs_container {
                                None => Vec::new(),
                                Some(trc) => {
                                    let mut refs: Vec<(u32, String)> = trc
                                        .children()
                                        .filter(|n| n.has_tag_name((NS, "trackRef")))
                                        .filter_map(|tr| {
                                            let track_ref = tr.attribute("ref")?.to_string();
                                            let sn: u32 = tr
                                                .attribute("sequenceNumber")
                                                .and_then(|v| v.parse().ok())
                                                .unwrap_or(0);
                                            Some((sn, track_ref))
                                        })
                                        .collect();
                                    refs.sort_by_key(|(sn, _)| *sn);
                                    refs
                                }
                            }
                        }
                    }
                }
            };

            points.push((seq, bip_id, track_ids));
        }

        points.sort_by_key(|(seq, _, _)| *seq);
        base_itineraries.insert(bi_id, points);
    }

    // -----------------------------------------------------------------------
    // Step 2: Parse every itinerary as an ordered list of ItineraryRange
    //         references into base itineraries (with optional BIP start/end).
    // -----------------------------------------------------------------------

    struct ItineraryRange {
        base_itinerary_ref: String,
        start_bip: Option<String>, // inclusive first BIP id
        end_bip: Option<String>,   // inclusive last BIP id
        seq: u32,
    }

    let mut itineraries: HashMap<String, Vec<ItineraryRange>> = HashMap::new();

    for iti in tt_node.descendants().filter(|n| n.has_tag_name((NS, "itinerary"))) {
        let iti_id = match iti.attribute("id") {
            Some(v) => v.to_string(),
            None => continue,
        };

        let mut ranges: Vec<ItineraryRange> = iti
            .children()
            .filter(|n| n.has_tag_name((NS, "range")))
            .filter_map(|r| {
                let base_ref = r.attribute("baseItineraryRef")?.to_string();
                let seq: u32 =
                    r.attribute("sequenceNumber").and_then(|v| v.parse().ok()).unwrap_or(0);
                Some(ItineraryRange {
                    base_itinerary_ref: base_ref,
                    start_bip: r.attribute("start").map(str::to_string),
                    end_bip: r.attribute("end").map(str::to_string),
                    seq,
                })
            })
            .collect();
        ranges.sort_by_key(|r| r.seq);
        itineraries.insert(iti_id, ranges);
    }

    // -----------------------------------------------------------------------
    // Step 3: Parse operationalTrain elements and resolve their routes.
    // -----------------------------------------------------------------------

    let mut routes: HashMap<String, Route> = HashMap::new();

    for ot in tt_node.descendants().filter(|n| n.has_tag_name((NS, "operationalTrain"))) {
        let ot_id = match ot.attribute("id") {
            Some(v) => v.to_string(),
            None => continue,
        };

        // Take the first operationalTrainVariant.
        let variant = match ot
            .children()
            .find(|n| n.has_tag_name((NS, "operationalTrainVariant")))
        {
            Some(v) => v,
            None => continue,
        };

        let itinerary_ref = match variant.attribute("itineraryRef") {
            Some(v) => v.to_string(),
            None => {
                eprintln!(
                    "Warning: operationalTrainVariant in train '{ot_id}' has no @itineraryRef — skipped"
                );
                continue;
            }
        };

        let iti_ranges = match itineraries.get(&itinerary_ref) {
            Some(r) => r,
            None => {
                eprintln!(
                    "Warning: itinerary '{itinerary_ref}' referenced by train '{ot_id}' not found — skipped"
                );
                continue;
            }
        };

        // Flatten all ranges into an ordered list of track IDs.
        let mut all_track_ids: Vec<String> = Vec::new();

        for range in iti_ranges {
            let bi_points = match base_itineraries.get(&range.base_itinerary_ref) {
                Some(p) => p,
                None => {
                    eprintln!(
                        "Warning: baseItinerary '{}' not found (train '{}') — skipped",
                        range.base_itinerary_ref, ot_id
                    );
                    continue;
                }
            };

            // Determine the BIP sequence-number window.
            let start_seq = range
                .start_bip
                .as_deref()
                .and_then(|id| bip_seq.get(id).copied())
                .unwrap_or(0);
            let end_seq = range
                .end_bip
                .as_deref()
                .and_then(|id| bip_seq.get(id).copied())
                .unwrap_or(u32::MAX);

            for (bip_seq_n, _bip_id, track_ids) in bi_points {
                if *bip_seq_n < start_seq || *bip_seq_n > end_seq {
                    continue;
                }
                // The last BIP of the range has no onward section — skip its
                // track list (which will be empty anyway, but be explicit).
                if *bip_seq_n == end_seq {
                    continue;
                }
                for (_, tid) in track_ids {
                    all_track_ids.push(tid.clone());
                }
            }
        }

        // Resolve track IDs → RouteElements via the infrastructure.
        let elements: Vec<RouteElement> = all_track_ids
            .into_iter()
            .filter_map(|tid| {
                let track = match infra.tracks.get(&tid) {
                    Some(t) => t,
                    None => {
                        eprintln!(
                            "Warning: track '{tid}' in route for train '{ot_id}' not found in infrastructure — skipped"
                        );
                        return None;
                    }
                };
                let ne = match infra.net_elements.get(&track.net_element_id) {
                    Some(ne) => ne,
                    None => {
                        eprintln!(
                            "Warning: netElement '{}' (track '{tid}', train '{ot_id}') not found in infrastructure — skipped",
                            track.net_element_id
                        );
                        return None;
                    }
                };
                Some(RouteElement {
                    track_id: tid,
                    net_element_id: ne.id.clone(),
                    length_m: ne.length_m,
                })
            })
            .collect();

        if elements.is_empty() {
            eprintln!(
                "Warning: train '{ot_id}' resolved to an empty route — no Route created"
            );
            continue;
        }

        let route = Route::new(elements);
        eprintln!(
            "Route for train '{ot_id}': {} elements, total {:.0} m",
            route.elements.len(),
            route.total_length_m,
        );
        routes.insert(ot_id, route);
    }

    Ok(routes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::model::{NetElement, Track};

    const NS_DECL: &str = r#"xmlns:rail3="https://www.railml.org/schemas/3.3""#;

    fn wrap(body: &str) -> String {
        format!(r#"<?xml version="1.0"?><rail3:railml {NS_DECL}>{body}</rail3:railml>"#)
    }

    /// Build a minimal Infrastructure with a given list of (track_id, ne_id, length_m) triples.
    fn make_infra(tracks: &[(&str, &str, f64)]) -> Infrastructure {
        let mut net_elements = HashMap::new();
        let mut track_map = HashMap::new();
        for &(tid, nid, len) in tracks {
            net_elements.insert(
                nid.to_string(),
                NetElement { id: nid.to_string(), length_m: len },
            );
            track_map.insert(
                tid.to_string(),
                Track { id: tid.to_string(), net_element_id: nid.to_string() },
            );
        }
        Infrastructure { net_elements, tracks: track_map, ops: HashMap::new() }
    }

    // Minimal XML helpers for building timetable documents.

    fn base_itinerary(id: &str, points: &str) -> String {
        format!(r#"<rail3:baseItinerary id="{id}">{points}</rail3:baseItinerary>"#)
    }

    fn bip(id: &str, seq: u32, followup: &str) -> String {
        format!(
            r#"<rail3:baseItineraryPoint id="{id}" sequenceNumber="{seq}" locationRef="OP_any">
                 {followup}
                 <rail3:stop/>
               </rail3:baseItineraryPoint>"#
        )
    }

    fn followup(track_refs: &str) -> String {
        format!(
            r#"<rail3:followupSections>
                 <rail3:followupSection>
                   <rail3:trackRefs>{track_refs}</rail3:trackRefs>
                 </rail3:followupSection>
               </rail3:followupSections>"#
        )
    }

    fn track_ref(r: &str, seq: u32) -> String {
        format!(r#"<rail3:trackRef ref="{r}" sequenceNumber="{seq}"/>"#)
    }

    fn itinerary(id: &str, ranges: &str) -> String {
        format!(r#"<rail3:itinerary id="{id}">{ranges}</rail3:itinerary>"#)
    }

    fn range(base_ref: &str, seq: u32) -> String {
        format!(r#"<rail3:range baseItineraryRef="{base_ref}" sequenceNumber="{seq}"/>"#)
    }

    fn range_sliced(base_ref: &str, seq: u32, start: &str, end: &str) -> String {
        format!(
            r#"<rail3:range baseItineraryRef="{base_ref}" sequenceNumber="{seq}" start="{start}" end="{end}"/>"#
        )
    }

    fn op_train(id: &str, variant_id: &str, iti_ref: &str) -> String {
        format!(
            r#"<rail3:operationalTrain id="{id}">
                 <rail3:operationalTrainVariant id="{variant_id}" itineraryRef="{iti_ref}" validityRef="V1"/>
               </rail3:operationalTrain>"#
        )
    }

    fn full_doc(base_itineraries: &str, itineraries: &str, op_trains: &str) -> String {
        wrap(&format!(
            r#"<rail3:timetable>
                 <rail3:baseItineraries>{base_itineraries}</rail3:baseItineraries>
                 <rail3:itineraries>{itineraries}</rail3:itineraries>
                 <rail3:operationalTrains>{op_trains}</rail3:operationalTrains>
               </rail3:timetable>"#
        ))
    }

    #[test]
    fn test_basic_two_bip_route() {
        // BIP_A → (track_A, track_B) → BIP_B
        let bi = base_itinerary(
            "BIT_1",
            &format!(
                "{} {}",
                bip("BIP_A", 1, &followup(&format!("{} {}", track_ref("track_A", 1), track_ref("track_B", 2)))),
                bip("BIP_B", 2, ""),
            ),
        );
        let iti = itinerary("ITI_1", &range("BIT_1", 1));
        let ot = op_train("OT_Express", "OTV_1", "ITI_1");
        let doc = full_doc(&bi, &iti, &ot);

        let infra = make_infra(&[("track_A", "ne_1", 1000.0), ("track_B", "ne_2", 500.0)]);
        let routes = parse_routes_xml(&doc, "test", &infra).unwrap();

        let route = routes.get("OT_Express").expect("route not found");
        assert_eq!(route.elements.len(), 2);
        assert_eq!(route.elements[0].track_id, "track_A");
        assert_eq!(route.elements[1].track_id, "track_B");
        assert!((route.total_length_m - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn test_track_ref_order_by_sequence_number() {
        // track_refs listed in reverse XML order but sequenceNumber is correct.
        let bi = base_itinerary(
            "BIT_1",
            &format!(
                "{} {}",
                bip(
                    "BIP_A", 1,
                    &followup(&format!(
                        // Intentionally reversed: seq=2 before seq=1
                        "{} {}",
                        track_ref("track_B", 2),
                        track_ref("track_A", 1),
                    )),
                ),
                bip("BIP_B", 2, ""),
            ),
        );
        let iti = itinerary("ITI_1", &range("BIT_1", 1));
        let ot = op_train("OT_1", "OTV_1", "ITI_1");
        let doc = full_doc(&bi, &iti, &ot);

        let infra = make_infra(&[("track_A", "ne_A", 100.0), ("track_B", "ne_B", 200.0)]);
        let routes = parse_routes_xml(&doc, "test", &infra).unwrap();

        let route = &routes["OT_1"];
        assert_eq!(route.elements[0].track_id, "track_A"); // seq 1 first
        assert_eq!(route.elements[1].track_id, "track_B"); // seq 2 second
    }

    #[test]
    fn test_bip_range_slicing() {
        // BaseItinerary has 3 BIPs (A→B→C) but itinerary range only uses A→B.
        let bi = base_itinerary(
            "BIT_1",
            &format!(
                "{} {} {}",
                bip("BIP_A", 1, &followup(&track_ref("track_A", 1))),
                bip("BIP_B", 2, &followup(&track_ref("track_B", 1))),
                bip("BIP_C", 3, ""),
            ),
        );
        let iti = itinerary("ITI_1", &range_sliced("BIT_1", 1, "BIP_A", "BIP_B"));
        let ot = op_train("OT_1", "OTV_1", "ITI_1");
        let doc = full_doc(&bi, &iti, &ot);

        let infra =
            make_infra(&[("track_A", "ne_A", 100.0), ("track_B", "ne_B", 200.0)]);
        let routes = parse_routes_xml(&doc, "test", &infra).unwrap();

        let route = &routes["OT_1"];
        // Only track_A should be in the route; BIP_B→BIP_C track is excluded.
        assert_eq!(route.elements.len(), 1);
        assert_eq!(route.elements[0].track_id, "track_A");
    }

    #[test]
    fn test_best_priority_followup_section_selected() {
        // Two followupSections: priority 1 (lower priority) and priority 0 (best).
        let followups = r#"<rail3:followupSections>
            <rail3:followupSection priority="1">
              <rail3:trackRefs>
                <rail3:trackRef ref="track_alt" sequenceNumber="1"/>
              </rail3:trackRefs>
            </rail3:followupSection>
            <rail3:followupSection priority="0">
              <rail3:trackRefs>
                <rail3:trackRef ref="track_best" sequenceNumber="1"/>
              </rail3:trackRefs>
            </rail3:followupSection>
          </rail3:followupSections>"#;
        let bi = base_itinerary(
            "BIT_1",
            &format!(
                "{} {}",
                format!(
                    r#"<rail3:baseItineraryPoint id="BIP_A" sequenceNumber="1" locationRef="OP_any">
                         {followups}<rail3:stop/></rail3:baseItineraryPoint>"#
                ),
                bip("BIP_B", 2, ""),
            ),
        );
        let iti = itinerary("ITI_1", &range("BIT_1", 1));
        let ot = op_train("OT_1", "OTV_1", "ITI_1");
        let doc = full_doc(&bi, &iti, &ot);

        let infra = make_infra(&[
            ("track_best", "ne_best", 100.0),
            ("track_alt", "ne_alt", 200.0),
        ]);
        let routes = parse_routes_xml(&doc, "test", &infra).unwrap();

        let route = &routes["OT_1"];
        assert_eq!(route.elements[0].track_id, "track_best");
    }

    #[test]
    fn test_missing_track_in_infra_skips_gracefully() {
        let bi = base_itinerary(
            "BIT_1",
            &format!(
                "{} {}",
                bip("BIP_A", 1, &followup(&format!("{} {}", track_ref("track_good", 1), track_ref("track_missing", 2)))),
                bip("BIP_B", 2, ""),
            ),
        );
        let iti = itinerary("ITI_1", &range("BIT_1", 1));
        let ot = op_train("OT_1", "OTV_1", "ITI_1");
        let doc = full_doc(&bi, &iti, &ot);

        // Only track_good is in the infrastructure.
        let infra = make_infra(&[("track_good", "ne_good", 300.0)]);
        let routes = parse_routes_xml(&doc, "test", &infra).unwrap();

        let route = &routes["OT_1"];
        assert_eq!(route.elements.len(), 1);
        assert_eq!(route.elements[0].track_id, "track_good");
    }

    #[test]
    fn test_no_timetable_section_returns_empty_map() {
        let doc = wrap("<rail3:infrastructure/>");
        let infra = make_infra(&[]);
        let routes = parse_routes_xml(&doc, "test", &infra).unwrap();
        assert!(routes.is_empty());
    }

    #[test]
    fn test_missing_itinerary_ref_skips_train() {
        // operationalTrainVariant has no @itineraryRef → train skipped, no error.
        let bi = base_itinerary("BIT_1", &format!("{} {}", bip("BIP_A", 1, ""), bip("BIP_B", 2, "")));
        let iti = itinerary("ITI_1", &range("BIT_1", 1));
        let bad_ot = wrap(&format!(
            r#"<rail3:timetable>
                 <rail3:baseItineraries>{bi}</rail3:baseItineraries>
                 <rail3:itineraries>{iti}</rail3:itineraries>
                 <rail3:operationalTrains>
                   <rail3:operationalTrain id="OT_bad">
                     <rail3:operationalTrainVariant id="OTV_1" validityRef="V1"/>
                   </rail3:operationalTrain>
                 </rail3:operationalTrains>
               </rail3:timetable>"#
        ));
        let infra = make_infra(&[]);
        let routes = parse_routes_xml(&bad_ot, "test", &infra).unwrap();
        assert!(routes.is_empty());
    }

    #[test]
    fn test_multi_range_itinerary_concatenates_tracks() {
        // Two base itineraries, one itinerary with two ranges.
        let bi1 = base_itinerary(
            "BIT_1",
            &format!(
                "{} {}",
                bip("BIP_A1", 1, &followup(&track_ref("track_A", 1))),
                bip("BIP_A2", 2, ""),
            ),
        );
        let bi2 = base_itinerary(
            "BIT_2",
            &format!(
                "{} {}",
                bip("BIP_B1", 1, &followup(&track_ref("track_B", 1))),
                bip("BIP_B2", 2, ""),
            ),
        );
        let iti = itinerary(
            "ITI_1",
            &format!("{} {}", range("BIT_1", 1), range("BIT_2", 2)),
        );
        let ot = op_train("OT_1", "OTV_1", "ITI_1");
        let doc = full_doc(
            &format!("{bi1}{bi2}"),
            &iti,
            &ot,
        );

        let infra = make_infra(&[("track_A", "ne_A", 100.0), ("track_B", "ne_B", 200.0)]);
        let routes = parse_routes_xml(&doc, "test", &infra).unwrap();

        let route = &routes["OT_1"];
        assert_eq!(route.elements.len(), 2);
        assert_eq!(route.elements[0].track_id, "track_A");
        assert_eq!(route.elements[1].track_id, "track_B");
        assert!((route.total_length_m - 300.0).abs() < 1e-9);
    }

    #[test]
    fn test_all_tracks_missing_from_infra_produces_no_route() {
        // All track IDs referenced by the route are absent from the infrastructure.
        // The empty-elements guard should fire and no route should be produced.
        let bi = base_itinerary(
            "BIT_1",
            &format!(
                "{} {}",
                bip("BIP_A", 1, &followup(&format!("{} {}", track_ref("track_X", 1), track_ref("track_Y", 2)))),
                bip("BIP_B", 2, ""),
            ),
        );
        let iti = itinerary("ITI_1", &range("BIT_1", 1));
        let ot = op_train("OT_1", "OTV_1", "ITI_1");
        let doc = full_doc(&bi, &iti, &ot);

        // Empty infrastructure — none of the track IDs can be resolved.
        let infra = make_infra(&[]);
        let routes = parse_routes_xml(&doc, "test", &infra).unwrap();

        // No route should be created when every track resolves to nothing.
        assert!(!routes.contains_key("OT_1"));
    }
}
