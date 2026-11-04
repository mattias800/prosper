// Private same-TU companion, included inside prosper::test.
#pragma once
inline VkPrimitiveTopology pipeline_topology_class(VkPrimitiveTopology topology) {
    // Dynamic topology is core, but unrestricted changes BETWEEN topology classes are not.
    // Retain the class on every device rather than depending on extended_dynamic_state3.
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
        return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    default:
        return topology; // points, patches, and unknown values stay distinct
    }
}
