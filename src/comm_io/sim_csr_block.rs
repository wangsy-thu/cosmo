use std::collections::{BTreeMap, HashMap};

use crate::types::graph_query::GraphQuery;
use crate::types::graph_serialize::{ByteEncodable, TopologyDecode, TopologyEncode};

/// A simple implementation of CSR block, by replacing the V(u64) with u64.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct CSRSimpleCommBlock {
    /// Total number of vertices in this CSR block.
    pub vertex_count: u64,

    /// List of vertices with their corresponding offsets.
    /// Each tuple contains a vertex and its offset in the neighbor list.
    pub vertex_list: Vec<(u64, u64)>,

    /// Flattened list of all neighbors for all vertices.
    /// The offsets in vertex_list determine which neighbors belong to which vertex.
    pub neighbor_list: Vec<u64>,

    /// HashMap for efficient vertex lookup.
    /// Maps vertex data to its index in the vertex_list for quick retrieval.
    pub(crate) vertex_index: HashMap<u64, usize>,
}

impl GraphQuery<u64, u64> for CSRSimpleCommBlock {
    /// Retrieves all neighbors of the specified vertex.
    /// Returns an empty vector if the vertex doesn't exist or is marked as tombstone.
    fn read_neighbor(&self, vertex_id: &u64) -> Vec<u64> {
        // Step 1: Locate the vertex in the index
        let vertex_idx_opt = self.vertex_index.get(vertex_id);

        // Check if the vertex exists in this block
        let vertex_list_idx = match vertex_idx_opt {
            None => {
                // Vertex not found in the index
                // For performance reasons, we don't do a full scan and just return empty
                return vec![];
            }
            Some(vertex_idx) => {
                *vertex_idx
            }
        };

        // Step 2: Determine the range of neighbors in neighbor_list using offsets
        // Get the start offset for this vertex's neighbors
        let neighbors_start = match self.vertex_list[vertex_list_idx].1.try_into() {
            Ok(offset_usize) => {
                offset_usize
            }
            Err(_) => {
                return vec![];
            }
        };

        // Convert vertex_count to usize for comparison
        let vertex_count_usize = match self.vertex_count.try_into() {
            Ok(count) => count,
            Err(_) => {
                return vec![];
            },
        };

        // Determine the end offset:
        // - For the last vertex, it's the end of neighbor_list
        // - For other vertices, it's the start offset of the next vertex
        let neighbors_end = if vertex_list_idx + 1 == vertex_count_usize {
            self.neighbor_list.len()
        } else {
            match self.vertex_list[vertex_list_idx + 1].1.try_into() {
                Ok(offset_usize) => {
                    offset_usize
                }
                Err(_) => {
                    return vec![];
                }
            }
        };

        // Step 3: Extract the neighbors and filter out tombstone vertices
        self.neighbor_list[neighbors_start..neighbors_end].to_vec()
            .into_iter() // Keep only non-tombstone neighbors
            .collect::<Vec<_>>()
    }

    /// Checks if a vertex exists in the graph and is not marked as deleted.
    /// Returns true if the vertex exists and is active, false otherwise.
    fn has_vertex(&self, vertex_id: &u64) -> bool {
        // Check if the vertex exists in the index
        self.vertex_index.contains_key(vertex_id)
    }

    /// Checks if there is an active edge from src_id to dst_id.
    /// Returns true if the edge exists and is active, false otherwise.
    fn has_edge(&self, src_id: &u64, dst_id: &u64) -> bool {
        self.read_neighbor(src_id).iter().any(
            |vertex| *vertex == *dst_id
        )
    }

    /// Returns a list of all vertices in the graph.
    fn vertex_list(&self) -> Vec<u64> {
        self.vertex_list.iter().map(
            |vertex| vertex.0.clone()
        ).collect::<Vec<_>>()
    }

    /// Returns a complete representation of the graph as a map.
    /// Each entry maps a vertex ID to its vertex object and list of neighbors.
    fn all(&self) -> BTreeMap<u64, (u64, Vec<u64>)> {
        // Initialize the result map
        let mut graph_map = BTreeMap::<u64, (u64, Vec<u64>)>::new();

        // Process each vertex in the index
        for (vertex_id, vertex_array_idx) in self.vertex_index.iter() {
            // Step 1: Get the vertex object
            let vertex = self.vertex_list[*vertex_array_idx].0.clone();

            // Step 2: Determine the range of neighbors in neighbor_list using offsets
            // Get the start offset for this vertex's neighbors
            let neighbors_start = match self.vertex_list[*vertex_array_idx].1.try_into() {
                Ok(offset_usize) => {
                    offset_usize
                }
                Err(_) => {
                    panic!("Cast Error.")
                }
            };

            // Convert vertex_count to usize for comparison
            let vertex_count_usize = match self.vertex_count.try_into() {
                Ok(count) => count,
                Err(_) => panic!("Usize cast error."),
            };

            // Determine the end offset
            let neighbors_end = if *vertex_array_idx + 1 == vertex_count_usize {
                self.neighbor_list.len()
            } else {
                match self.vertex_list[*vertex_array_idx + 1].1.try_into() {
                    Ok(offset_usize) => {
                        offset_usize
                    }
                    Err(_) => {
                        panic!("Cast Error.");
                    }
                }
            };

            // Step 3: Extract all neighbors for this vertex
            let neighbor_list = self.neighbor_list[neighbors_start..neighbors_end].to_vec();

            // Add this vertex and its neighbors to the map
            graph_map.insert(*vertex_id, (vertex, neighbor_list));
        }

        graph_map
    }
}

#[allow(dead_code)]
impl TopologyDecode for CSRSimpleCommBlock  {
    /// Implements the TopologyDecode trait for CSRSimpleCommBlock.
    /// This method deserializes a byte array back into a CSRCommBlock structure.
    /// Returns None if the byte array is invalid or lacks sufficient data.
    fn from_bytes_topology(bytes: &[u8]) -> Option<Self> {
        let mut decode_offset = 0usize;

        // Ensure there are enough bytes to read the vertex count
        if decode_offset + u64::byte_size() > bytes.len() {
            // Insufficient data to decode vertex count
            return None;
        }

        // First decode the total number of vertices
        let vertex_count = u64::from_bytes(
            &bytes[decode_offset..decode_offset + u64::byte_size()]
        ).unwrap();
        decode_offset += u64::byte_size();

        // Initialize containers for decoded data
        let mut vertex_list = Vec::<(u64, u64)>::new();
        let mut vertex_index = HashMap::<u64, usize>::new();

        // Convert vertex_count to usize for iteration
        let vertex_count_usize = match vertex_count.try_into() {
            Ok(count) => count,
            Err(_) => return None,
        };

        // Decode each vertex and its offset
        for vertex_idx in 0..vertex_count_usize {
            // Decode the vertex ID.
            let vertex_id = u64::from_bytes(
                &bytes[decode_offset..decode_offset + u64::byte_size()]
            ).unwrap();
            decode_offset += u64::byte_size();
            // Add vertex to index map for quick lookups
            vertex_index.insert(vertex_id, vertex_idx);

            let offset_type_size = u64::byte_size();
            let mut offset_bytes = Vec::new();
            offset_bytes.resize(u64::byte_size(), 0u8);
            offset_bytes.copy_from_slice(&bytes[decode_offset..decode_offset + offset_type_size]);
            let decoded_vertex_offset = u64::from_bytes(&offset_bytes).unwrap();
            vertex_list.push((vertex_id, decoded_vertex_offset));
            decode_offset += offset_type_size;
        }

        // Decode the neighbor list until we reach the end of the byte array
        let mut neighbor_list = Vec::<u64>::new();
        loop {
            // Check if we've reached the end of the byte array
            if decode_offset >= bytes.len() {
                break;
            }

            // Check if there are enough bytes left to decode a vertex
            let decoded_end = decode_offset + u64::byte_size();
            if decoded_end > bytes.len() {
                // Partial data encountered - cannot decode complete vertex
                return None
            }

            // Decode the neighbor vertex
            let neighbor_opt = u64::from_bytes(
                &bytes[decode_offset..decoded_end]
            );
            match neighbor_opt {
                None => {
                    // Failed to decode neighbor
                    return None
                }
                Some(neighbor) => {
                    neighbor_list.push(neighbor);
                    decode_offset += u64::byte_size();
                }
            }
        }

        // Construct and return the complete CSRCommBlock
        Some(CSRSimpleCommBlock {
            vertex_count,
            vertex_list,
            neighbor_list,
            vertex_index,
        })
    }
}

#[allow(dead_code)]
impl TopologyEncode for CSRSimpleCommBlock {
    /// Implements the TopologyEncode trait for CSRCommBlock.
    /// This method serializes the graph structure into a byte vector.
    /// Note: This only encodes the topology information and omits the vertex_index HashMap.
    fn encode_topology(&self) -> Vec<u8> {
        let mut encoded_bytes = Vec::<u8>::new();

        // First encode the total number of vertices
        let vertex_count_bytes = &self.vertex_count.to_bytes();
        encoded_bytes.extend_from_slice(vertex_count_bytes);

        // Then encode each vertex and its corresponding offset in the neighbor list
        for (vertex, offset) in &self.vertex_list {
            encoded_bytes.extend_from_slice(&vertex.to_bytes());
            encoded_bytes.extend_from_slice(&offset.to_bytes());
        }

        // Finally encode all neighbors in the neighbor list
        for neighbor in &self.neighbor_list {
            encoded_bytes.extend_from_slice(&neighbor.to_bytes());
        }
        // The vertex_index HashMap is not encoded as it can be reconstructed from vertex_list

        encoded_bytes
    }
}

#[cfg(test)]
mod test_simple_csr_block {
    use std::collections::HashMap;

    use super::*;

    // Helper function to create a simple CSRCommBlock for testing.
    fn create_test_block() -> CSRSimpleCommBlock {
        // Create vertices
        let v1 = 1u64;
        let v2 = 2u64;
        let v3 = 3u64;

        // Create vertex list with their offsets in the neighbor list
        let vertex_list = vec![(v1, 0u64), (v2, 2u64), (v3, 5u64)];

        // Create neighbor list: v1 -> [v2, v3], v2 -> [v1, v3, v1], v3 -> [v2]
        let n1 = 2u64;  // v1's first neighbor is v2
        let n2 = 3u64;  // v1's second neighbor is v3
        let n3 = 1u64;  // v2's first neighbor is v1
        let n4 = 3u64;  // v2's second neighbor is v3
        let n5 = 1u64;  // v2's third neighbor is v1 (duplicated intentionally)
        let n6 = 2u64;  // v3's first neighbor is v2

        let neighbor_list = vec![n1, n2, n3, n4, n5, n6];

        // Create vertex index map
        let mut vertex_index = HashMap::new();
        vertex_index.insert(1, 0); // v1 is at index 0 in vertex_list
        vertex_index.insert(2, 1); // v2 is at index 1 in vertex_list
        vertex_index.insert(3, 2); // v3 is at index 2 in vertex_list

        CSRSimpleCommBlock {
            vertex_count: 3,
            vertex_list,
            neighbor_list,
            vertex_index,
        }
    }

    // Test encoding and decoding with a simple block.
    #[test]
    fn test_encode_decode_simple_block() {
        let original_block = create_test_block();

        // Encode the block
        let encoded = original_block.encode_topology();

        // Decode the block
        let decoded_block_opt = CSRSimpleCommBlock::from_bytes_topology(&encoded);
        assert!(decoded_block_opt.is_some(), "Failed to decode the block");

        let decoded_block = decoded_block_opt.unwrap();

        // Verify the vertex count
        assert_eq!(decoded_block.vertex_count, original_block.vertex_count);

        // Verify the vertex list length
        assert_eq!(decoded_block.vertex_list.len(), original_block.vertex_list.len());

        // Verify each vertex and its offset
        for i in 0..original_block.vertex_list.len() {
            let (orig_vertex, orig_offset) = original_block.vertex_list[i];
            let (decoded_vertex, decoded_offset) = decoded_block.vertex_list[i];

            assert_eq!(decoded_vertex, orig_vertex);
            assert_eq!(decoded_offset, orig_offset);
        }

        // Verify the neighbor list
        assert_eq!(decoded_block.neighbor_list.len(), original_block.neighbor_list.len());

        for i in 0..original_block.neighbor_list.len() {
            let orig_neighbor = original_block.neighbor_list[i];
            let decoded_neighbor = decoded_block.neighbor_list[i];

            assert_eq!(decoded_neighbor, orig_neighbor);
        }

        // Verify the vertex index
        assert_eq!(decoded_block.vertex_index.len(), original_block.vertex_index.len());

        for (key, _) in &original_block.vertex_index {
            assert!(decoded_block.vertex_index.contains_key(key));
            assert_eq!(decoded_block.vertex_index.get(key), original_block.vertex_index.get(key));
        }
    }

    // Test with an empty neighbor list
    #[test]
    fn test_encode_decode_empty_neighbors() {
        // Create a block with vertices but no edges
        let v1 = 1u64;
        let v2 = 2u64;

        let vertex_list = vec![(v1, 0u64), (v2, 064)];
        let neighbor_list = vec![];

        let mut vertex_index = HashMap::new();
        vertex_index.insert(1, 0);
        vertex_index.insert(2, 1);

        let original_block = CSRSimpleCommBlock {
            vertex_count: 2u64,
            vertex_list,
            neighbor_list,
            vertex_index,
        };

        // Encode and decode
        let encoded = original_block.encode_topology();
        let decoded_block_opt = CSRSimpleCommBlock::from_bytes_topology(&encoded);

        assert!(decoded_block_opt.is_some());
        let decoded_block = decoded_block_opt.unwrap();

        // Verify
        assert_eq!(decoded_block.vertex_count, 2);
        assert_eq!(decoded_block.vertex_list.len(), 2);
        assert_eq!(decoded_block.neighbor_list.len(), 0);
        assert_eq!(decoded_block.vertex_index.len(), 2);
    }

    // Test with a large number of vertices and neighbors
    #[test]
    fn test_encode_decode_large_block() {
        const VERTEX_COUNT: usize = 100;
        const NEIGHBORS_PER_VERTEX: usize = 5;

        // Create vertices and their offsets
        let mut vertex_list = Vec::with_capacity(VERTEX_COUNT);
        let mut vertex_index = HashMap::new();

        for i in 0..VERTEX_COUNT {
            let vertex = i as u64;
            let offset = (i * NEIGHBORS_PER_VERTEX) as u64;
            vertex_list.push((vertex, offset));
            vertex_index.insert(i as u64, i);
        }

        // Create neighbor list
        let mut neighbor_list = Vec::with_capacity(VERTEX_COUNT * NEIGHBORS_PER_VERTEX);

        for i in 0..VERTEX_COUNT {
            for j in 0..NEIGHBORS_PER_VERTEX {
                // Connect to the next 5 vertices (wrapping around if needed)
                let neighbor_id = ((i + j + 1) % VERTEX_COUNT) as u64;
                neighbor_list.push(neighbor_id);
            }
        }

        let original_block = CSRSimpleCommBlock {
            vertex_count: VERTEX_COUNT as u64,
            vertex_list,
            neighbor_list,
            vertex_index,
        };

        // Encode and decode
        let encoded = original_block.encode_topology();
        let decoded_block_opt = CSRSimpleCommBlock::from_bytes_topology(&encoded);

        assert!(decoded_block_opt.is_some());
        let decoded_block = decoded_block_opt.unwrap();

        // Verify the basics
        assert_eq!(decoded_block.vertex_count, VERTEX_COUNT as u64);
        assert_eq!(decoded_block.vertex_list.len(), VERTEX_COUNT);
        assert_eq!(decoded_block.neighbor_list.len(), VERTEX_COUNT * NEIGHBORS_PER_VERTEX);

        // Verify some random vertices and their edges
        for test_index in [0, 10, 25, 50, 75, 99] {
            if test_index < VERTEX_COUNT {
                let (orig_vertex, orig_offset) = &original_block.vertex_list[test_index];
                let (decoded_vertex, decoded_offset) = &decoded_block.vertex_list[test_index];

                assert_eq!(*decoded_vertex, *orig_vertex);
                assert_eq!(*decoded_offset, *orig_offset);

                // Check neighbors
                let start_idx = *orig_offset as usize;
                let end_idx = if test_index == VERTEX_COUNT - 1 {
                    original_block.neighbor_list.len()
                } else {
                    original_block.vertex_list[test_index + 1].1 as usize
                };

                for i in start_idx..end_idx {
                    let orig_neighbor = &original_block.neighbor_list[i];
                    let decoded_neighbor = &decoded_block.neighbor_list[i];

                    assert_eq!(*decoded_neighbor, *orig_neighbor);
                }
            }
        }
    }

    // Test read_neighbor functionality
    #[test]
    fn test_read_neighbor() {
        let block = create_test_block();

        // Test reading neighbors of vertex 1
        let neighbors_of_1 = block.read_neighbor(&1);
        assert_eq!(neighbors_of_1.len(), 2);
        assert!(neighbors_of_1.iter().any(|v| *v == 2));
        assert!(neighbors_of_1.iter().any(|v| *v == 3));

        // Test reading neighbors of vertex 2
        let neighbors_of_2 = block.read_neighbor(&2);
        assert_eq!(neighbors_of_2.len(), 3);
        assert_eq!(neighbors_of_2[0], 1);
        assert_eq!(neighbors_of_2[1], 3);
        assert_eq!(neighbors_of_2[2], 1); // Duplicated neighbor

        // Test reading neighbors of vertex 3
        let neighbors_of_3 = block.read_neighbor(&3);
        assert_eq!(neighbors_of_3.len(), 1);
        assert_eq!(neighbors_of_3[0], 2);

        // Test reading neighbors of non-existent vertex
        let neighbors_of_4 = block.read_neighbor(&4);
        assert_eq!(neighbors_of_4.len(), 0);
    }

    // Test has_vertex functionality
    #[test]
    fn test_has_vertex() {
        let block = create_test_block();

        // Test existing vertices
        assert!(block.has_vertex(&1));
        assert!(block.has_vertex(&2));
        assert!(block.has_vertex(&3));

        // Test non-existent vertex
        assert!(!block.has_vertex(&4));
        assert!(!block.has_vertex(&0));
    }

    // Test has_edge functionality
    #[test]
    fn test_has_edge() {
        let block = create_test_block();

        println!("1's neighbor: {:?}", block.read_neighbor(&1));
        // Test existing edges
        assert!(block.has_edge(&1, &2));
        assert!(block.has_edge(&1, &3));
        assert!(block.has_edge(&2, &1));
        assert!(block.has_edge(&2, &3));
        assert!(block.has_edge(&3, &2));

        // Test non-existent edges
        assert!(!block.has_edge(&3, &1));
        assert!(!block.has_edge(&4, &1));
        assert!(!block.has_edge(&1, &4));
        assert!(!block.has_edge(&1, &1)); // No self-loop for vertex 1
        assert!(!block.has_edge(&3, &3)); // No self-loop for vertex 3
    }

    // Test vertex_list functionality
    #[test]
    fn test_vertex_list() {
        let block = create_test_block();

        let vertices = block.vertex_list();

        // Check length
        assert_eq!(vertices.len(), 3);

        // Check all vertex IDs are present
        let vertex_ids: Vec<u64> = vertices.iter().map(|v| v).cloned().collect();
        assert!(vertex_ids.contains(&1));
        assert!(vertex_ids.contains(&2));
        assert!(vertex_ids.contains(&3));

        // Check the order is preserved from vertex_list
        assert_eq!(vertices[0], 1);
        assert_eq!(vertices[1], 2);
        assert_eq!(vertices[2], 3);
    }

    // Test all functionality
    #[test]
    fn test_all() {
        let block = create_test_block();

        let graph = block.all();

        // Check graph size
        assert_eq!(graph.len(), 3);

        // Check vertices and their neighbors
        let (v1, neighbors_of_1) = graph.get(&1).unwrap();
        assert_eq!(*v1, 1);
        assert_eq!(neighbors_of_1.len(), 2);
        assert_eq!(neighbors_of_1[0], 2);
        assert_eq!(neighbors_of_1[1], 3);

        let (v2, neighbors_of_2) = graph.get(&2).unwrap();
        assert_eq!(*v2, 2);
        assert_eq!(neighbors_of_2.len(), 3);
        assert_eq!(neighbors_of_2[0], 1);
        assert_eq!(neighbors_of_2[1], 3);
        assert_eq!(neighbors_of_2[2], 1); // Duplicated neighbor

        let (v3, neighbors_of_3) = graph.get(&3).unwrap();
        assert_eq!(*v3, 3);
        assert_eq!(neighbors_of_3.len(), 1);
        assert_eq!(neighbors_of_3[0], 2);
    }

    // Test with empty graph (no edges)
    #[test]
    fn test_query_empty_graph() {
        // Create a block with vertices but no edges
        let v1 = 1u64;
        let v2 = 2u64;

        let vertex_list = vec![(v1, 0u64), (v2, 0u64)];
        let neighbor_list = vec![];

        let mut vertex_index = HashMap::new();
        vertex_index.insert(1, 0);
        vertex_index.insert(2, 1);

        let block = CSRSimpleCommBlock {
            vertex_count: 2u64,
            vertex_list,
            neighbor_list,
            vertex_index,
        };

        // Test read_neighbor
        assert_eq!(block.read_neighbor(&1).len(), 0);
        assert_eq!(block.read_neighbor(&2).len(), 0);

        // Test has_vertex
        assert!(block.has_vertex(&1));
        assert!(block.has_vertex(&2));

        // Test has_edge
        assert!(!block.has_edge(&1, &2));
        assert!(!block.has_edge(&2, &1));

        // Test vertex_list
        let vertices = block.vertex_list();
        assert_eq!(vertices.len(), 2);
        assert_eq!(vertices[0], 1);
        assert_eq!(vertices[1], 2);

        // Test all
        let graph = block.all();
        assert_eq!(graph.len(), 2);
        assert_eq!(graph.get(&1).unwrap().1.len(), 0); // No neighbors
        assert_eq!(graph.get(&2).unwrap().1.len(), 0); // No neighbors
    }

    // Test with a large number of edges from a single vertex
    #[test]
    fn test_query_hub_vertex() {
        // Create a "hub" vertex with many connections
        const NEIGHBOR_COUNT: usize = 100;

        let hub = 1u64;
        let vertex_list = vec![(hub, 0u64)];

        let mut neighbor_list = Vec::with_capacity(NEIGHBOR_COUNT);
        let mut vertex_index = HashMap::new();
        vertex_index.insert(1, 0);

        // Create 100 neighbors for the hub
        for i in 2..=NEIGHBOR_COUNT+1 {
            neighbor_list.push(i as u64);
        }

        let block = CSRSimpleCommBlock {
            vertex_count: 1u64,
            vertex_list,
            neighbor_list,
            vertex_index,
        };

        // Test read_neighbor for the hub
        let hub_neighbors = block.read_neighbor(&1);
        assert_eq!(hub_neighbors.len(), NEIGHBOR_COUNT);

        // Test has_edge for all connections
        for i in 2..=NEIGHBOR_COUNT+1 {
            assert!(block.has_edge(&1, &(i as u64)));
        }

        // Test all
        let graph = block.all();
        assert_eq!(graph.len(), 1);
        assert_eq!(graph.get(&1).unwrap().1.len(), NEIGHBOR_COUNT);
    }

    // Test with a cyclic graph
    #[test]
    fn test_query_cyclic_graph() {
        // Create a simple cycle: 1 -> 2 -> 3 -> 1
        let v1 = 1u64;
        let v2 = 2u64;
        let v3 = 3u64;

        let vertex_list = vec![(v1, 0u64), (v2, 1u64), (v3, 2u64)];
        let neighbor_list = vec![
            2u64,  // v1 -> v2
            3u64,  // v2 -> v3
            1u64,  // v3 -> v1
        ];

        let mut vertex_index = HashMap::new();
        vertex_index.insert(1, 0);
        vertex_index.insert(2, 1);
        vertex_index.insert(3, 2);

        let block = CSRSimpleCommBlock {
            vertex_count: 3u64,
            vertex_list,
            neighbor_list,
            vertex_index,
        };

        // Test has_edge for the cycle
        assert!(block.has_edge(&1, &2));
        assert!(block.has_edge(&2, &3));
        assert!(block.has_edge(&3, &1));

        // Test all
        let graph = block.all();

        // Follow the cycle in the generated graph
        let v1_neighbors = &graph.get(&1).unwrap().1;
        assert_eq!(v1_neighbors.len(), 1);
        assert_eq!(v1_neighbors[0], 2);

        let v2_neighbors = &graph.get(&2).unwrap().1;
        assert_eq!(v2_neighbors.len(), 1);
        assert_eq!(v2_neighbors[0], 3);

        let v3_neighbors = &graph.get(&3).unwrap().1;
        assert_eq!(v3_neighbors.len(), 1);
        assert_eq!(v3_neighbors[0], 1);
    }
}