use std::collections::{BTreeMap, HashMap};
use crate::types::CSRGraph;
use crate::types::graph_query::GraphQuery;
use crate::types::graph_serialize::{Offset, TopologyDecode, TopologyEncode, VertexId, VertexLength};
use crate::types::Vertex;

/// A block contains a CSR (Compressed Sparse Row) format representation of a part of a graph.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct CSRCommBlock<T, L, O> {
    /// Total number of vertices in this CSR block.
    pub vertex_count: L,

    /// List of vertices with their corresponding offsets.
    /// Each tuple contains a vertex and its offset in the neighbor list.
    pub vertex_list: Vec<(Vertex<T>, O)>,

    /// Flattened list of all neighbors for all vertices.
    /// The offsets in vertex_list determine which neighbors belong to which vertex.
    pub neighbor_list: Vec<Vertex<T>>,

    /// HashMap for efficient vertex lookup.
    /// Maps vertex data to its index in the vertex_list for quick retrieval.
    pub(crate) vertex_index: HashMap<T, usize>,
}

#[allow(dead_code)]
impl CSRCommBlock<u64, u64, u64> {
    /// Converts this CSR block into a complete CSR graph structure.
    /// This method creates a new CSR graph containing only the internal connections
    /// between vertices present in this block.
    pub fn generate_csr_graph(&self) -> CSRGraph<u64, u64, u64> {
        // Vector to store the offset of each vertex in the new edge list
        let mut inner_offset = Vec::new();
        // Vector to store the flattened list of neighboring vertices
        let mut inner_edge_list = Vec::new();
        // Tracks the current offset position as we build the edge list
        let mut current_offset = 0u64;

        // Process each vertex in this block
        for (vertex, _) in &self.vertex_list {
            let global_vertex_id = vertex.vertex_id;

            // Filter neighbors to include only those present in this block
            // and convert them to their local indices
            let mut inner_neighbors =
                self.read_neighbor(&global_vertex_id).iter().filter_map(|n| {
                    if self.vertex_index.contains_key(&n.vertex_id) {
                        Some(self.vertex_index.get(&n.vertex_id).unwrap().clone() as u64)
                    } else {
                        None
                    }
                }).collect::<Vec<_>>();

            // Record the offset for this vertex
            inner_offset.push(current_offset);
            // Update the current offset for the next vertex
            current_offset += inner_neighbors.len() as u64;
            // Add this vertex's neighbors to the edge list
            inner_edge_list.append(&mut inner_neighbors);
        }

        // Use the same vertex count as the block
        let new_vertex_count = self.vertex_count;

        // Construct and return the new CSR graph
        CSRGraph {
            vertex_count: new_vertex_count,
            offsets: inner_offset,
            neighbor_list: inner_edge_list,
            community_index: Default::default(),
        }
    }
}

#[allow(dead_code)]
impl<T, L, O> TopologyEncode for CSRCommBlock<T, L, O>
where
    T: VertexId,      // Type T must implement the VertexId trait
    L: VertexLength,  // Type L must implement the VertexLength trait
    O: Offset         // Type O must implement the Offset trait
{
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
            encoded_bytes.extend_from_slice(&vertex.encode_topology());
            encoded_bytes.extend_from_slice(&offset.to_bytes());
        }

        // Finally encode all neighbors in the neighbor list
        for neighbor in &self.neighbor_list {
            encoded_bytes.extend_from_slice(&neighbor.encode_topology());
        }
        // The vertex_index HashMap is not encoded as it can be reconstructed from vertex_list

        encoded_bytes
    }
}

#[allow(dead_code)]
impl<T, L, O> TopologyDecode for CSRCommBlock<T, L, O>
where
    T: VertexId,      // Type T must implement the VertexId trait
    L: VertexLength,  // Type L must implement the VertexLength trait
    O: Offset         // Type O must implement the Offset trait
{
    /// Implements the TopologyDecode trait for CSRCommBlock.
    /// This method deserializes a byte array back into a CSRCommBlock structure.
    /// Returns None if the byte array is invalid or lacks sufficient data.
    fn from_bytes_topology(bytes: &[u8]) -> Option<Self> {
        let mut decode_offset = 0usize;

        // Ensure there are enough bytes to read the vertex count
        if decode_offset + L::byte_size() > bytes.len() {
            // Insufficient data to decode vertex count
            return None;
        }

        // First decode the total number of vertices
        let vertex_count = L::from_bytes(
            &bytes[decode_offset..decode_offset + L::byte_size()]
        ).unwrap();
        decode_offset += L::byte_size();

        // Initialize containers for decoded data
        let mut vertex_list = Vec::<(Vertex<T>, O)>::new();
        let mut vertex_index = HashMap::<T, usize>::new();

        // Convert vertex_count to usize for iteration
        let vertex_count_usize = match vertex_count.try_into() {
            Ok(count) => count,
            Err(_) => return None,
        };

        // Decode each vertex and its offset
        for vertex_idx in 0..vertex_count_usize {
            // Decode the vertex
            let decoded_vertex_opt = Vertex::<T>::from_bytes_topology(
                &bytes[decode_offset..decode_offset + Vertex::<T>::byte_size()]
            );
            match decoded_vertex_opt {
                None => {
                    // Failed to decode vertex
                    return None
                }
                Some(decoded_vertex) => {
                    // Add vertex to index map for quick lookups
                    vertex_index.insert(decoded_vertex.vertex_id, vertex_idx);

                    // Decode the vertex's offset
                    decode_offset += Vertex::<T>::byte_size();
                    let offset_type_size = O::byte_size();
                    let mut offset_bytes = Vec::new();
                    offset_bytes.resize(O::byte_size(), 0u8);
                    offset_bytes.copy_from_slice(&bytes[decode_offset..decode_offset + offset_type_size]);
                    let decoded_vertex_offset = O::from_bytes(&offset_bytes).unwrap();
                    vertex_list.push((decoded_vertex, decoded_vertex_offset));
                    decode_offset += O::byte_size();
                }
            }
        }

        // Decode the neighbor list until we reach the end of the byte array
        let mut neighbor_list = Vec::<Vertex<T>>::new();
        loop {
            // Check if we've reached the end of the byte array
            if decode_offset >= bytes.len() {
                break;
            }

            // Check if there are enough bytes left to decode a vertex
            let decoded_end = decode_offset + Vertex::<T>::byte_size();
            if decoded_end > bytes.len() {
                // Partial data encountered - cannot decode complete vertex
                return None
            }

            // Decode the neighbor vertex
            let neighbor_opt = Vertex::<T>::from_bytes_topology(
                &bytes[decode_offset..decoded_end]
            );
            match neighbor_opt {
                None => {
                    // Failed to decode neighbor
                    return None
                }
                Some(neighbor) => {
                    neighbor_list.push(neighbor);
                    decode_offset += Vertex::<T>::byte_size();
                }
            }
        }

        // Construct and return the complete CSRCommBlock
        Some(CSRCommBlock {
            vertex_count,
            vertex_list,
            neighbor_list,
            vertex_index,
        })
    }
}

impl<T, L, O> GraphQuery<T, Vertex<T>> for CSRCommBlock<T, L, O>
where
    T: VertexId,      // Type T must implement the VertexId trait
    L: VertexLength,  // Type L must implement the VertexLength trait
    O: Offset         // Type O must implement the Offset trait
{
    /// Retrieves all neighbors of the specified vertex.
    /// Returns an empty vector if the vertex doesn't exist or is marked as tombstone.
    fn read_neighbor(&self, vertex_id: &T) -> Vec<Vertex<T>> {
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

        // If the vertex is marked as deleted (tomb), return empty
        if self.vertex_list[vertex_list_idx].0.tomb == 1 {
            return vec![]
        }

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
            .into_iter()
            .filter(|vertex| vertex.tomb == 0)  // Keep only non-tombstone neighbors
            .collect::<Vec<_>>()
    }

    /// Checks if a vertex exists in the graph and is not marked as deleted.
    /// Returns true if the vertex exists and is active, false otherwise.
    fn has_vertex(&self, vertex_id: &T) -> bool {
        // Check if the vertex exists in the index
        if !self.vertex_index.contains_key(vertex_id) {
            return false;
        }

        // Check if the vertex is marked as deleted (tomb)
        let vertex_array_idx = self.vertex_index.get(vertex_id).unwrap();
        let vertex = self.vertex_list[*vertex_array_idx].0;
        if vertex.tomb != 0 {
            return false;
        }

        true
    }

    /// Checks if there is an active edge from src_id to dst_id.
    /// Returns true if the edge exists and is active, false otherwise.
    fn has_edge(&self, src_id: &T, dst_id: &T) -> bool {
        self.read_neighbor(src_id).iter().any(
            |vertex| vertex.vertex_id == *dst_id && vertex.tomb == 0
        )
    }

    /// Returns a list of all vertices in the graph.
    fn vertex_list(&self) -> Vec<Vertex<T>> {
        self.vertex_list.iter().map(
            |vertex| vertex.0.clone()
        ).collect::<Vec<_>>()
    }

    /// Returns a complete representation of the graph as a map.
    /// Each entry maps a vertex ID to its vertex object and list of neighbors.
    fn all(&self) -> BTreeMap<T, (Vertex<T>, Vec<Vertex<T>>)> {
        // Initialize the result map
        let mut graph_map = BTreeMap::<T, (Vertex<T>, Vec<Vertex<T>>)>::new();

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

#[cfg(test)]
mod test_csr_block {
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    // Helper function to create a vertex with a given id.
    fn create_vertex(id: u32) -> Vertex<u32> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Vertex {
            vertex_id: id,
            timestamp: current_time,
            tomb: 0,
        }
    }

    // Helper function to create a simple CSRCommBlock for testing.
    fn create_test_block() -> CSRCommBlock<u32, u32, u32> {
        // Create vertices
        let v1 = create_vertex(1);
        let v2 = create_vertex(2);
        let v3 = create_vertex(3);

        // Create vertex list with their offsets in the neighbor list
        let vertex_list = vec![(v1, 0u32), (v2, 2u32), (v3, 5u32)];

        // Create neighbor list: v1 -> [v2, v3], v2 -> [v1, v3, v1], v3 -> [v2]
        let n1 = create_vertex(2);  // v1's first neighbor is v2
        let n2 = create_vertex(3);  // v1's second neighbor is v3
        let n3 = create_vertex(1);  // v2's first neighbor is v1
        let n4 = create_vertex(3);  // v2's second neighbor is v3
        let n5 = create_vertex(1);  // v2's third neighbor is v1 (duplicated intentionally)
        let n6 = create_vertex(2);  // v3's first neighbor is v2

        let neighbor_list = vec![n1, n2, n3, n4, n5, n6];

        // Create vertex index map
        let mut vertex_index = HashMap::new();
        vertex_index.insert(1, 0); // v1 is at index 0 in vertex_list
        vertex_index.insert(2, 1); // v2 is at index 1 in vertex_list
        vertex_index.insert(3, 2); // v3 is at index 2 in vertex_list

        CSRCommBlock {
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
        let decoded_block_opt = CSRCommBlock::<u32, u32, u32>::from_bytes_topology(&encoded);
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

            assert_eq!(decoded_vertex.vertex_id, orig_vertex.vertex_id);
            assert_eq!(decoded_vertex.timestamp, orig_vertex.timestamp);
            assert_eq!(decoded_vertex.tomb, orig_vertex.tomb);
            assert_eq!(decoded_offset, orig_offset);
        }

        // Verify the neighbor list
        assert_eq!(decoded_block.neighbor_list.len(), original_block.neighbor_list.len());

        for i in 0..original_block.neighbor_list.len() {
            let orig_neighbor = original_block.neighbor_list[i];
            let decoded_neighbor = decoded_block.neighbor_list[i];

            assert_eq!(decoded_neighbor.vertex_id, orig_neighbor.vertex_id);
            assert_eq!(decoded_neighbor.timestamp, orig_neighbor.timestamp);
            assert_eq!(decoded_neighbor.tomb, orig_neighbor.tomb);
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
        let v1 = create_vertex(1);
        let v2 = create_vertex(2);

        let vertex_list = vec![(v1, 0u32), (v2, 032)];
        let neighbor_list = vec![];

        let mut vertex_index = HashMap::new();
        vertex_index.insert(1, 0);
        vertex_index.insert(2, 1);

        let original_block = CSRCommBlock {
            vertex_count: 2u32,
            vertex_list,
            neighbor_list,
            vertex_index,
        };

        // Encode and decode
        let encoded = original_block.encode_topology();
        let decoded_block_opt = CSRCommBlock::<u32, u32, u32>::from_bytes_topology(&encoded);

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
            let vertex = create_vertex(i as u32);
            let offset = (i * NEIGHBORS_PER_VERTEX) as u32;
            vertex_list.push((vertex, offset));
            vertex_index.insert(i as u32, i);
        }

        // Create neighbor list
        let mut neighbor_list = Vec::with_capacity(VERTEX_COUNT * NEIGHBORS_PER_VERTEX);

        for i in 0..VERTEX_COUNT {
            for j in 0..NEIGHBORS_PER_VERTEX {
                // Connect to the next 5 vertices (wrapping around if needed)
                let neighbor_id = ((i + j + 1) % VERTEX_COUNT) as u32;
                neighbor_list.push(create_vertex(neighbor_id));
            }
        }

        let original_block = CSRCommBlock {
            vertex_count: VERTEX_COUNT as u32,
            vertex_list,
            neighbor_list,
            vertex_index,
        };

        // Encode and decode
        let encoded = original_block.encode_topology();
        let decoded_block_opt = CSRCommBlock::<u32, u32, u32>::from_bytes_topology(&encoded);

        assert!(decoded_block_opt.is_some());
        let decoded_block = decoded_block_opt.unwrap();

        // Verify the basics
        assert_eq!(decoded_block.vertex_count, VERTEX_COUNT as u32);
        assert_eq!(decoded_block.vertex_list.len(), VERTEX_COUNT);
        assert_eq!(decoded_block.neighbor_list.len(), VERTEX_COUNT * NEIGHBORS_PER_VERTEX);

        // Verify some random vertices and their edges
        for test_index in [0, 10, 25, 50, 75, 99] {
            if test_index < VERTEX_COUNT {
                let (orig_vertex, orig_offset) = &original_block.vertex_list[test_index];
                let (decoded_vertex, decoded_offset) = &decoded_block.vertex_list[test_index];

                assert_eq!(decoded_vertex.vertex_id, orig_vertex.vertex_id);
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

                    assert_eq!(decoded_neighbor.vertex_id, orig_neighbor.vertex_id);
                }
            }
        }
    }

    // Test with vertices that have tomb values set
    #[test]
    fn test_encode_decode_with_tombstones() {
        // Create a block with some vertices marked as deleted
        let mut v1 = create_vertex(1);
        let v2 = create_vertex(2);
        let mut v3 = create_vertex(3);

        // Mark v1 and v3 as deleted (tombstone)
        v1.tomb = 1;  // Deleted
        v3.tomb = 2;  // Force deleted

        let vertex_list = vec![(v1, 0u32), (v2, 1u32), (v3, 3u32)];

        // v2 has edges to both deleted vertices and a self-loop
        let n1 = create_vertex(1);  // Edge to deleted vertex
        let n2 = create_vertex(2);  // Self-loop
        let n3 = create_vertex(3);  // Edge to force deleted vertex

        let neighbor_list = vec![n1, n2, n3];

        let mut vertex_index = HashMap::new();
        vertex_index.insert(1, 0);
        vertex_index.insert(2, 1);
        vertex_index.insert(3, 2);

        let original_block = CSRCommBlock {
            vertex_count: 3u32,
            vertex_list,
            neighbor_list,
            vertex_index,
        };

        // Encode and decode
        let encoded = original_block.encode_topology();
        let decoded_block_opt = CSRCommBlock::<u32, u32, u32>::from_bytes_topology(&encoded);

        assert!(decoded_block_opt.is_some());
        let decoded_block = decoded_block_opt.unwrap();

        // Verify the tomb values are preserved
        assert_eq!(decoded_block.vertex_list[0].0.tomb, 1);  // v1 should be marked deleted
        assert_eq!(decoded_block.vertex_list[1].0.tomb, 0);  // v2 should be marked active
        assert_eq!(decoded_block.vertex_list[2].0.tomb, 2);  // v3 should be marked force deleted

        // Verify neighbor relationships
        assert_eq!(decoded_block.neighbor_list[0].vertex_id, 1);
        assert_eq!(decoded_block.neighbor_list[1].vertex_id, 2);
        assert_eq!(decoded_block.neighbor_list[2].vertex_id, 3);
    }

    // Test handling of invalid input data
    #[test]
    fn test_decode_invalid_data() {
        // Test with empty byte array
        let empty_result = CSRCommBlock::<u32, u32, u32>::from_bytes_topology(&[]);
        assert!(empty_result.is_none());

        // Test with incomplete vertex count
        let incomplete_vertex_count = vec![1, 0, 0];  // Only 3 bytes for u32
        let result = CSRCommBlock::<u32, u32, u32>::from_bytes_topology(&incomplete_vertex_count);
        assert!(result.is_none());

        // Create valid data then corrupt it
        let valid_block = create_test_block();
        let mut valid_encoded = valid_block.encode_topology();

        // Corrupt the encoded data by truncating it
        valid_encoded.truncate(valid_encoded.len() / 2);

        // Try to decode the corrupted data
        let corrupted_result = CSRCommBlock::<u32, u32, u32>::from_bytes_topology(&valid_encoded);

        // This might actually succeed with partial data, depending on your implementation
        // What's important is that it doesn't crash or produce invalid data
        if corrupted_result.is_some() {
            let decoded = corrupted_result.unwrap();
            // The vertex count should match, but the lists will be shorter
            assert_eq!(decoded.vertex_count, valid_block.vertex_count);
            assert!(decoded.neighbor_list.len() < valid_block.neighbor_list.len());
        }
    }

    // Test with different data types for T and L
    #[test]
    fn test_different_data_types() {
        // Test with u64 for vertex IDs
        let v1 = Vertex {
            vertex_id: 1u64,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            tomb: 0,
        };

        let v2 = Vertex {
            vertex_id: 2u64,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            tomb: 0,
        };

        let vertex_list = vec![(v1, 0u32), (v2, 1u32)];
        let neighbor_list = vec![v2];  // v1 points to v2

        let mut vertex_index = HashMap::new();
        vertex_index.insert(1u64, 0);
        vertex_index.insert(2u64, 1);

        let original_block = CSRCommBlock {
            vertex_count: 2u16,  // Using u16 for L
            vertex_list,
            neighbor_list,
            vertex_index,
        };

        // Encode and decode
        let encoded = original_block.encode_topology();
        let decoded_block_opt = CSRCommBlock::<u64, u16, u32>::from_bytes_topology(&encoded);

        assert!(decoded_block_opt.is_some());
        let decoded_block = decoded_block_opt.unwrap();

        // Verify
        assert_eq!(decoded_block.vertex_count, 2u16);
        assert_eq!(decoded_block.vertex_list.len(), 2);
        assert_eq!(decoded_block.vertex_list[0].0.vertex_id, 1u64);
        assert_eq!(decoded_block.vertex_list[1].0.vertex_id, 2u64);
        assert_eq!(decoded_block.neighbor_list.len(), 1);
        assert_eq!(decoded_block.neighbor_list[0].vertex_id, 2u64);
    }

    // Add these tests to the existing test_csr_block module

    // Test read_neighbor functionality
    #[test]
    fn test_read_neighbor() {
        let block = create_test_block();

        // Test reading neighbors of vertex 1
        let neighbors_of_1 = block.read_neighbor(&1);
        assert_eq!(neighbors_of_1.len(), 2);
        assert!(neighbors_of_1.iter().any(|v| v.vertex_id == 2));
        assert!(neighbors_of_1.iter().any(|v| v.vertex_id == 3));

        // Test reading neighbors of vertex 2
        let neighbors_of_2 = block.read_neighbor(&2);
        assert_eq!(neighbors_of_2.len(), 3);
        assert_eq!(neighbors_of_2[0].vertex_id, 1);
        assert_eq!(neighbors_of_2[1].vertex_id, 3);
        assert_eq!(neighbors_of_2[2].vertex_id, 1); // Duplicated neighbor

        // Test reading neighbors of vertex 3
        let neighbors_of_3 = block.read_neighbor(&3);
        assert_eq!(neighbors_of_3.len(), 1);
        assert_eq!(neighbors_of_3[0].vertex_id, 2);

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
        let vertex_ids: Vec<u32> = vertices.iter().map(|v| v.vertex_id).collect();
        assert!(vertex_ids.contains(&1));
        assert!(vertex_ids.contains(&2));
        assert!(vertex_ids.contains(&3));

        // Check the order is preserved from vertex_list
        assert_eq!(vertices[0].vertex_id, 1);
        assert_eq!(vertices[1].vertex_id, 2);
        assert_eq!(vertices[2].vertex_id, 3);
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
        assert_eq!(v1.vertex_id, 1);
        assert_eq!(neighbors_of_1.len(), 2);
        assert_eq!(neighbors_of_1[0].vertex_id, 2);
        assert_eq!(neighbors_of_1[1].vertex_id, 3);

        let (v2, neighbors_of_2) = graph.get(&2).unwrap();
        assert_eq!(v2.vertex_id, 2);
        assert_eq!(neighbors_of_2.len(), 3);
        assert_eq!(neighbors_of_2[0].vertex_id, 1);
        assert_eq!(neighbors_of_2[1].vertex_id, 3);
        assert_eq!(neighbors_of_2[2].vertex_id, 1); // Duplicated neighbor

        let (v3, neighbors_of_3) = graph.get(&3).unwrap();
        assert_eq!(v3.vertex_id, 3);
        assert_eq!(neighbors_of_3.len(), 1);
        assert_eq!(neighbors_of_3[0].vertex_id, 2);
    }

    // Test with empty graph (no edges)
    #[test]
    fn test_query_empty_graph() {
        // Create a block with vertices but no edges
        let v1 = create_vertex(1);
        let v2 = create_vertex(2);

        let vertex_list = vec![(v1, 0u32), (v2, 0u32)];
        let neighbor_list = vec![];

        let mut vertex_index = HashMap::new();
        vertex_index.insert(1, 0);
        vertex_index.insert(2, 1);

        let block = CSRCommBlock {
            vertex_count: 2u32,
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
        assert_eq!(vertices[0].vertex_id, 1);
        assert_eq!(vertices[1].vertex_id, 2);

        // Test all
        let graph = block.all();
        assert_eq!(graph.len(), 2);
        assert_eq!(graph.get(&1).unwrap().1.len(), 0); // No neighbors
        assert_eq!(graph.get(&2).unwrap().1.len(), 0); // No neighbors
    }

    // Test with deleted vertices (tomb values)
    #[test]
    fn test_query_with_tombstones() {
        // Create a block with some vertices marked as deleted
        let mut v1 = create_vertex(1);
        let v2 = create_vertex(2);
        let mut v3 = create_vertex(3);

        // Mark v1 and v3 as deleted (tombstone)
        v1.tomb = 1;  // Deleted
        v3.tomb = 2;  // Force deleted

        let vertex_list = vec![(v1, 0u32), (v2, 1u32), (v3, 2u32)];

        // v2 has edges to both deleted vertices and a self-loop
        let n1 = create_vertex(1);  // Edge to deleted vertex
        let n2 = create_vertex(2);  // Self-loop
        let n3 = create_vertex(3);  // Edge to force deleted vertex

        let neighbor_list = vec![n1, n2, n3];

        let mut vertex_index = HashMap::new();
        vertex_index.insert(1, 0);
        vertex_index.insert(2, 1);
        vertex_index.insert(3, 2);

        let block = CSRCommBlock {
            vertex_count: 3u32,
            vertex_list,
            neighbor_list,
            vertex_index,
        };

        // The Query trait should still work with deleted vertices

        // Test has_vertex (should return true even for deleted vertices)
        assert!(!block.has_vertex(&1));
        assert!(!block.has_vertex(&3));

        // Test read_neighbor
        let neighbors_of_2 = block.read_neighbor(&2);
        assert_eq!(neighbors_of_2.len(), 1);

        // Test all includes deleted vertices
        let graph = block.all();
        assert_eq!(graph.len(), 3);

        // Verify tomb values in the generated graph
        assert_eq!(graph.get(&1).unwrap().0.tomb, 1);
        assert_eq!(graph.get(&3).unwrap().0.tomb, 2);
    }

    // Test with a large number of edges from a single vertex
    #[test]
    fn test_query_hub_vertex() {
        // Create a "hub" vertex with many connections
        const NEIGHBOR_COUNT: usize = 100;

        let hub = create_vertex(1);
        let vertex_list = vec![(hub, 0u32)];

        let mut neighbor_list = Vec::with_capacity(NEIGHBOR_COUNT);
        let mut vertex_index = HashMap::new();
        vertex_index.insert(1, 0);

        // Create 100 neighbors for the hub
        for i in 2..=NEIGHBOR_COUNT+1 {
            neighbor_list.push(create_vertex(i as u32));
        }

        let block = CSRCommBlock {
            vertex_count: 1u32,
            vertex_list,
            neighbor_list,
            vertex_index,
        };

        // Test read_neighbor for the hub
        let hub_neighbors = block.read_neighbor(&1);
        assert_eq!(hub_neighbors.len(), NEIGHBOR_COUNT);

        // Test has_edge for all connections
        for i in 2..=NEIGHBOR_COUNT+1 {
            assert!(block.has_edge(&1, &(i as u32)));
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
        let v1 = create_vertex(1);
        let v2 = create_vertex(2);
        let v3 = create_vertex(3);

        let vertex_list = vec![(v1, 0u32), (v2, 1u32), (v3, 2u32)];
        let neighbor_list = vec![
            create_vertex(2),  // v1 -> v2
            create_vertex(3),  // v2 -> v3
            create_vertex(1),  // v3 -> v1
        ];

        let mut vertex_index = HashMap::new();
        vertex_index.insert(1, 0);
        vertex_index.insert(2, 1);
        vertex_index.insert(3, 2);

        let block = CSRCommBlock {
            vertex_count: 3u32,
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
        assert_eq!(v1_neighbors[0].vertex_id, 2);

        let v2_neighbors = &graph.get(&2).unwrap().1;
        assert_eq!(v2_neighbors.len(), 1);
        assert_eq!(v2_neighbors[0].vertex_id, 3);

        let v3_neighbors = &graph.get(&3).unwrap().1;
        assert_eq!(v3_neighbors.len(), 1);
        assert_eq!(v3_neighbors[0].vertex_id, 1);
    }
}
