/// Helpers for navigating and extracting typed values from `roxmltree` documents.
///
/// All functions accept a `ctx` string that is prepended to every error message so
/// callers don't have to repeat location context on every attribute access.

/// Parse a typed attribute from `node`.
///
/// Returns `Err` if the attribute is absent or its value cannot be parsed into `T`.
pub fn parse_attr<T: std::str::FromStr>(
    node: roxmltree::Node,
    attr_name: &str,
    ctx: &str,
) -> Result<T, String> {
    let raw = node
        .attribute(attr_name)
        .ok_or_else(|| format!("{ctx}: missing attribute '{attr_name}'"))?;
    raw.parse()
        .map_err(|_| format!("{ctx}: invalid value for '{attr_name}': {raw:?}"))
}

/// Find a required direct child element with the given namespace and tag.
pub fn required_child<'a, 'input>(
    node: roxmltree::Node<'a, 'input>,
    ns: &str,
    tag: &str,
    ctx: &str,
) -> Result<roxmltree::Node<'a, 'input>, String> {
    node.children()
        .find(|n| n.has_tag_name((ns, tag)))
        .ok_or_else(|| format!("{ctx}: no <{tag}> child element"))
}

/// Find a required descendant element (any depth) with the given namespace and tag.
pub fn required_descendant<'a, 'input>(
    node: roxmltree::Node<'a, 'input>,
    ns: &str,
    tag: &str,
    ctx: &str,
) -> Result<roxmltree::Node<'a, 'input>, String> {
    node.descendants()
        .find(|n| n.has_tag_name((ns, tag)))
        .ok_or_else(|| format!("{ctx}: no <{tag}> descendant element"))
}
