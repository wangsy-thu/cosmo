use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use indicatif::{ProgressBar, ProgressStyle};
use crate::config::READ_BUFFER_SIZE;
use crate::types::graph_query::GraphQuery;
use crate::types::graph_serialize::{Offset, TopologyDecode, TopologyEncode, VertexId, VertexLength};

pub(crate) mod graph_serialize;
pub(crate) mod graph_query;

// Define the Vertex struct (10 Bytes each, but 12 Bytes when applying size_of).
#[derive(Debug, Clone, Copy)]
pub struct Vertex<T> {
    /// Unique identifier for this vertex in the system, typically u32
    pub vertex_id: T,
    /// Timestamp when the vertex was created, defaults to current time
    pub(crate) timestamp: u64,
    /// Deletion marker: 0 = exists, 1 = deleted, 2 = force deleted
    pub(crate) tomb: u8
}

#[allow(dead_code)]
impl<T> Vertex<T>
where
    T: VertexId
{
    /// Returns the size in bytes that this vertex type occupies in memory
    /// Calculated as size of generic type T plus 9 bytes (8 for timestamp, 1 for tomb)
    pub fn byte_size() -> usize {
        T::byte_size() + 9
    }
}

impl<T> TopologyEncode for Vertex<T>
where
    T: VertexId
{
    /// Serializes the vertex data into a byte vector
    /// Returns a vector containing the binary representation of this vertex
    fn encode_topology(&self) -> Vec<u8> {
        let mut result = Vec::new();
        // Serialize the vertex_id field using its implementation
        result.extend_from_slice(&self.vertex_id.to_bytes());
        // Serialize the timestamp as little-endian bytes
        result.extend_from_slice(&self.timestamp.to_le_bytes());
        // Serialize the tomb field as a single byte
        result.push(self.tomb);
        result
    }
}

impl<T> TopologyDecode for Vertex<T>
where
    T: VertexId
{
    /// Deserializes vertex data from a byte slice
    /// Returns Some(Vertex) if deserialization succeeds, None otherwise
    fn from_bytes_topology(bytes: &[u8]) -> Option<Self> {
        // Calculate required byte length and verify we have enough data
        let required_size = T::byte_size() + 8 + 1; // vertex_id + timestamp + tomb
        if bytes.len() < required_size {
            return None;
        }

        // Deserialize vertex_id from the first section of bytes
        let vertex_id = T::from_bytes(&bytes[0..T::byte_size()])?;

        // Deserialize timestamp (8 bytes) from the middle section
        let timestamp_start = T::byte_size();
        let timestamp_end = timestamp_start + 8;
        let mut timestamp_bytes = [0u8; 8];
        timestamp_bytes.copy_from_slice(&bytes[timestamp_start..timestamp_end]);
        let timestamp = u64::from_le_bytes(timestamp_bytes);

        // Deserialize tomb (1 byte) from the last section
        let tomb = bytes[timestamp_end];

        // Construct and return the vertex
        Some(Vertex {
            vertex_id,
            timestamp,
            tomb,
        })
    }
}

/// A CSR (Compressed Sparse Row) implementation of a graph, storing complete graph structure.
/// Optimized for high-performance by omitting timestamp and tombstone management.
/// Assumes vertex IDs are continuous integers without gaps.
///
/// # Type Parameters
/// - `T`: Type for vertex IDs (typically u32 or similar)
/// - `L`: Type for vertex count (typically a numeric type)
/// - `O`: Type for offset values in the CSR structure
///
/// # Performance Notes
/// This implementation prioritizes memory efficiency and traversal speed over
/// flexibility for modifications.
#[allow(dead_code)]
#[derive(Debug)]
pub struct CSRGraph<T, L, O> {
    /// Total number of vertices in the graph
    pub vertex_count: L,

    /// Offset array that indicates where each vertex's adjacency list begins
    /// For each vertex i, its neighbors are stored in neighbor_list[offsets[i] to offsets[i+1]]
    pub offsets: Vec<O>,

    /// Flattened adjacency list containing all neighbors of all vertices
    /// Segmented according to the offsets array
    pub neighbor_list: Vec<T>,

    /// Maps vertex IDs to their community assignments
    /// Each vertex belongs to a community identified by u32
    pub community_index: BTreeMap<T, u32>,
}


/// Implementation of the GraphQuery trait for CSRGraph
/// Provides methods to query graph structure efficiently
impl<T, L, O> GraphQuery<T, T> for CSRGraph<T, L, O>
where
    T: VertexId,     // Type representing vertex IDs
    L: VertexLength, // Type representing count of vertices
    O: Offset        // Type representing offsets in the CSR structure
{
    /// Retrieves all neighbors of a specified vertex
    ///
    /// # Arguments
    /// * `vertex_id` - Reference to the ID of the vertex whose neighbors we want
    ///
    /// # Returns
    /// * `Vec<T>` - A vector containing the IDs of all neighboring vertices
    ///              Returns empty vector if vertex doesn't exist or has no neighbors
    fn read_neighbor(&self, vertex_id: &T) -> Vec<T> {
        // Convert vertex_id to usize for array indexing
        let vertex_id_usize: usize = match TryInto::<usize>::try_into(*vertex_id) {
            Ok(id) => id,
            Err(_) => return vec![], // Return empty vector if conversion fails
        };

        // Convert vertex_count to usize for comparison
        let vertex_count_usize: usize = match TryInto::<usize>::try_into(self.vertex_count) {
            Ok(count) => count,
            Err(_) => return vec![], // Return empty vector if conversion fails
        };

        // Check if vertex_id is within valid range
        if vertex_id_usize >= vertex_count_usize {
            vec![] // Return empty vector for out-of-range vertex IDs
        } else {
            // Get starting position in neighbor_list for this vertex
            let start_offset_usize: usize = match TryInto::<usize>::try_into(self.offsets[vertex_id_usize]) {
                Ok(offset) => offset,
                Err(_) => return vec![], // Return empty vector if conversion fails
            };

            // Get ending position in neighbor_list for this vertex
            let end_offset_usize: usize = if vertex_id_usize + 1 < self.offsets.len() {
                match TryInto::<usize>::try_into(self.offsets[vertex_id_usize + 1]) {
                    Ok(offset) => offset,
                    Err(_) => return vec![], // Return empty vector if conversion fails
                }
            } else {
                // If this is the last vertex, use the end of neighbor_list
                self.neighbor_list.len()
            };

            // Early return if there are no neighbors (equal offsets)
            if start_offset_usize == end_offset_usize {
                return vec![];
            }

            // Extract and return the slice of neighbors for this vertex
            self.neighbor_list[start_offset_usize..end_offset_usize].to_vec()
        }
    }

    /// Checks if a vertex exists in the graph
    ///
    /// # Arguments
    /// * `vertex_id` - Reference to the ID of the vertex to check
    ///
    /// # Returns
    /// * `bool` - True if the vertex exists, false otherwise
    fn has_vertex(&self, vertex_id: &T) -> bool {
        // Convert vertex_id to usize for array indexing
        let vertex_id_usize: usize = match TryInto::<usize>::try_into(*vertex_id) {
            Ok(id) => id,
            Err(_) => return false, // Return false if conversion fails
        };

        // Convert vertex_count to usize for comparison
        let vertex_count_usize: usize = match TryInto::<usize>::try_into(self.vertex_count) {
            Ok(count) => count,
            Err(_) => return false, // Return false if conversion fails
        };

        // Check if vertex_id is within valid range
        vertex_id_usize < vertex_count_usize // Note: this is the correct logic (< not >=)
    }

    /// Checks if an edge exists between two vertices
    ///
    /// # Arguments
    /// * `src_id` - Reference to the ID of the source vertex
    /// * `dst_id` - Reference to the ID of the destination vertex
    ///
    /// # Returns
    /// * `bool` - True if the edge exists, false otherwise
    fn has_edge(&self, src_id: &T, dst_id: &T) -> bool {
        if self.has_vertex(src_id) {
            // Check if dst_id appears in the neighbor list of src_id
            self.read_neighbor(src_id).iter().any(|&vertex_id| {
                vertex_id == *dst_id
            })
        } else {
            false // Source vertex doesn't exist, so edge can't exist
        }
    }

    /// Returns a list of all vertex IDs in the graph
    /// This method is marked as "not commonly used" and may be less optimized
    ///
    /// # Returns
    /// * `Vec<T>` - A vector containing all vertex IDs
    fn vertex_list(&self) -> Vec<T> {
        let vertex_count_usize: usize = match TryInto::<usize>::try_into(self.vertex_count) {
            Ok(count) => count,
            Err(_) => return vec![], // Return empty vector if conversion fails
        };

        // Generate sequential vertex IDs from 0 to vertex_count-1
        // Filtering out any ID that can't be converted to type T
        (0..vertex_count_usize)
            .filter_map(|i| T::try_from(i).ok())
            .collect()
    }

    /// Returns a complete representation of the graph as a map of vertices to their neighbors
    ///
    /// This method:
    /// 1. Retrieves all vertices in the graph
    /// 2. For each vertex, obtains its neighbor list
    /// 3. Builds a map that associates each vertex with itself and its neighbors
    ///
    /// # Returns
    /// * `BTreeMap<T, (T, Vec<T>)>` - A map where:
    ///   - The key is the vertex ID
    ///   - The value is a tuple containing:
    ///     - The vertex ID again (for convenience)
    ///     - A vector of all neighboring vertex IDs
    ///
    /// This provides a complete view of the graph structure that can be
    /// easily iterated over or used for graph analysis algorithms.
    fn all(&self) -> BTreeMap<T, (T, Vec<T>)> {
        // Get the list of all vertices in the graph
        let vertex_list = self.vertex_list();

        // Create a new map to store the complete graph information
        let mut all_graph_info = BTreeMap::<T, (T, Vec<T>)>::new();

        // For each vertex, retrieve its neighbors and add to the map
        for vertex in vertex_list.into_iter() {
            let neighbor_list = self.read_neighbor(&vertex);
            all_graph_info.insert(vertex, (vertex, neighbor_list));
        }

        // Return the complete graph structure
        all_graph_info
    }
}

#[allow(dead_code)]
impl CSRGraph<u64, u64, u64> {
    /// Loads a graph from a graph file in a specific format
    ///
    /// # Format
    /// The file should have the following structure:
    /// - First line: Contains metadata with format "? [vertex_count] [edge_count]"
    /// - Vertex lines: Start with "v", followed by vertex ID and optional community ID
    ///   Format: "v [vertex_id] ... [community_id]" (community_id at position 3)
    /// - Edge lines: Start with "e", followed by source and destination vertex IDs
    ///   Format: "e [source_id] [destination_id]"
    ///
    /// # Arguments
    /// * `file_path` - Path to the graph file
    ///
    /// # Returns
    /// * `CSRGraph<u64, u64, u64>` - A CSR representation of the graph
    ///
    /// # Panics
    /// * If the file cannot be opened or read
    /// * If the file format is incorrect
    /// * If parsing of numeric values fails
    pub fn from_graph_file(file_path: &str) -> CSRGraph<u64, u64, u64> {
        // Open the graph file with a buffered reader for efficient reading
        let graph_file = File::open(file_path).unwrap();
        let mut graph_reader = BufReader::with_capacity(READ_BUFFER_SIZE, graph_file);
        let mut first_line = String::new();
        graph_reader.read_line(&mut first_line).unwrap();

        // Parse the first line to extract vertex and edge counts
        let first_line_tokens: Vec<&str> = first_line.split_whitespace().collect();
        assert_eq!(first_line_tokens.len(), 3);
        let vertex_count = first_line_tokens[1].parse::<usize>().unwrap();
        let edge_count = first_line_tokens[2].parse::<usize>().unwrap();

        // Pre-allocate space for the neighbor list and vertex degrees
        let mut neighbor_list = Vec::with_capacity(edge_count);
        let mut degrees = vec![0u64; vertex_count];

        // Map to store community assignments for vertices
        let mut community_index = BTreeMap::<u64, u32>::new();

        // Setup thread-safe progress bar for user feedback
        let pb = Arc::new(ProgressBar::new((vertex_count + edge_count) as u64));
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
            .unwrap()
            .progress_chars("=>-"));
        pb.set_message("Graph Loading.");

        // Process each line in the file
        for line in graph_reader.lines() {
            if let Ok(line) = line {
                // Split the line into tokens
                let tokens: Vec<&str> = line.split_whitespace().collect();

                if tokens[0] == "v" {
                    // Process vertex line
                    let parsed_vid = tokens[1].parse::<u64>().ok().expect("File format error.");

                    // If community information exists (at index 3), store it
                    if tokens.len() >= 4 {
                        let community_id = tokens[3].parse::<u32>().ok().expect("File format error.");
                        community_index.insert(parsed_vid, community_id);
                    }
                } else if tokens[0] == "e" {
                    // Process edge line
                    let src = tokens[1].parse::<u64>().ok().expect("File format error.");
                    let dst = tokens[2].parse::<u64>().ok().expect("File format error.");

                    // Add destination to neighbor list and increment source vertex degree
                    neighbor_list.push(dst);
                    degrees[src as usize] += 1;
                }
                pb.inc(1);
            }
        }

        // Compute CSR offset array from vertex degrees
        // Each offset indicates where the adjacency list of a vertex begins
        let mut offsets = vec![0u64; vertex_count];
        for v in 0..vertex_count - 1 {
            offsets[v + 1] = offsets[v] + degrees[v];
        }

        // Create and return the CSRGraph instance
        let vertex_count_u64 = vertex_count as u64;
        Self {
            vertex_count: vertex_count_u64,
            offsets,
            neighbor_list,
            community_index,
        }
    }
}

/// A subgraph in-memory representation for graph computing.
/// It will be used in building SCC-DAG (Strongly Connected Components Directed Acyclic Graph).
#[allow(dead_code)]
#[derive(Debug)]
pub struct CSRSubGraph<T, L, O> {
    /// Total number of vertices in the graph
    pub vertex_count: L,

    /// List of vertices with their associated data
    /// Each entry is a tuple containing the vertex ID (T) and its metadata (O)
    pub vertex_list: Vec<(T, O)>,

    /// Flattened adjacency list containing all neighbors of all vertices
    /// Segmented according to the offsets stored in vertex_index
    /// This implements the Compressed Sparse Row (CSR) format for efficient storage
    pub neighbor_list: Vec<T>,

    /// Maps vertex IDs to their offsets in the neighbor_list
    /// Used to quickly locate the neighbors of a specific vertex
    pub vertex_index: HashMap<T, usize>,
}

#[allow(dead_code)]
impl GraphQuery<u64, u64> for CSRSubGraph<u64, u64, u64> {
    /// Retrieves all neighbors (connected vertices) for a given vertex ID
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph instance
    /// * `vertex_id` - Reference to the ID of the vertex whose neighbors we want to find
    ///
    /// # Returns
    ///
    /// * `Vec<u64>` - A vector containing the IDs of all neighboring vertices
    ///
    fn read_neighbor(&self, vertex_id: &u64) -> Vec<u64> {
        // Step 1: Locate the vertex in the index to get its position in the vertex list
        let vertex_opt = self.vertex_index.get(vertex_id);

        // Step 2: Check if the vertex exists in this graph
        let vertex_list_idx = match vertex_opt {
            None => {
                // Vertex not found in the index
                // For performance reasons, we don't perform a full scan and simply return an empty vector
                return vec![];
            }
            Some(vertex_idx) => {
                *vertex_idx
            }
        };

        // Step 3: Determine the range of neighbors in neighbor_list using offsets
        // Get the starting position for this vertex's neighbors in the neighbor list
        let neighbors_start = self.vertex_list[vertex_list_idx].1 as usize;

        // Determine the ending position:
        // - For the last vertex in the list, use the end of neighbor_list
        // - For other vertices, use the start position of the next vertex
        let neighbors_end = if vertex_list_idx + 1 == self.vertex_count as usize {
            self.neighbor_list.len()
        } else {
            self.vertex_list[vertex_list_idx + 1].1 as usize
        };

        // Step 4: Extract the neighbors within the determined range
        // Returns a vector containing all neighboring vertex IDs
        self.neighbor_list[neighbors_start..neighbors_end].to_vec()
    }

    /// Checks whether a vertex with the given ID exists in the graph
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph instance
    /// * `vertex_id` - Reference to the vertex ID to check
    ///
    /// # Returns
    ///
    /// * `bool` - Returns true if the vertex exists, false otherwise
    ///
    fn has_vertex(&self, vertex_id: &u64) -> bool {
        // Simply check if the vertex_id exists as a key in the vertex_index hashmap
        // This provides O(1) constant time lookup performance
        self.vertex_index.contains_key(vertex_id)
    }

    /// Checks whether an edge exists between two vertices in the graph
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph instance
    /// * `src_id` - Reference to the source vertex ID
    /// * `dst_id` - Reference to the destination vertex ID
    ///
    /// # Returns
    ///
    /// * `bool` - Returns true if an edge exists from src_id to dst_id, false otherwise
    ///
    fn has_edge(&self, src_id: &u64, dst_id: &u64) -> bool {
        // First retrieve all neighbors of the source vertex
        // Then check if the destination vertex ID exists in the neighbor list
        // Using the any() iterator adapter for efficient short-circuit evaluation
        self.read_neighbor(src_id).iter().any(
            |vertex| *vertex == *dst_id
        )
    }

    /// Returns a vector containing all vertex IDs in the graph
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph instance
    ///
    /// # Returns
    ///
    /// * `Vec<u64>` - A vector containing all vertex IDs in the graph
    ///
    fn vertex_list(&self) -> Vec<u64> {
        // Iterate through the vertex_list, extracting only the vertex IDs (the first element of each tuple)
        // Transform each tuple (vertex_id, offset) into just the vertex_id using map()
        // Finally, collect the results into a new Vec<u64>
        self.vertex_list.iter().map(
            |vertex| vertex.0.clone()
        ).collect::<Vec<_>>()
    }

    /// Returns a complete representation of the graph as a BTreeMap
    ///
    /// # Arguments
    ///
    /// * `&self` - Reference to the graph instance
    ///
    /// # Returns
    ///
    /// * `BTreeMap<u64, (u64, Vec<u64>)>` - A map where:
    ///   - The key is the vertex ID
    ///   - The value is a tuple containing:
    ///     - The vertex ID (duplicated from the key)
    ///     - A vector of all neighboring vertex IDs
    ///
    fn all(&self) -> BTreeMap<u64, (u64, Vec<u64>)> {
        // Initialize an empty BTreeMap to store the complete graph structure
        // Using BTreeMap for ordered traversal based on vertex IDs
        let mut graph_map =
            BTreeMap::<u64, (u64, Vec<u64>)>::new();

        // Iterate through all vertices in the index
        for (&vertex_id, &vertex_list_idx) in &self.vertex_index {
            // Step 1: Determine the range of neighbors in neighbor_list using offsets
            // Get the start offset for this vertex's neighbors
            let neighbors_start = self.vertex_list[vertex_list_idx].1 as usize;

            // Determine the end offset:
            // - If this is the last vertex, use the end of neighbor_list
            // - Otherwise, use the start offset of the next vertex
            let neighbors_end = if vertex_list_idx + 1 == self.vertex_count as usize {
                self.neighbor_list.len()
            } else {
                self.vertex_list[vertex_list_idx + 1].1 as usize
            };

            // Step 2: Extract all neighbors for this vertex into a new vector
            let neighbor_list = self.neighbor_list[neighbors_start..neighbors_end].to_vec();

            // Step 3: Add this vertex and its neighbors to the map
            // Format: vertex_id => (vertex_id, [neighbor_ids])
            graph_map.insert(vertex_id, (vertex_id, neighbor_list));
        }

        // Return the completed graph map
        graph_map
    }
}

#[allow(dead_code)]
impl CSRGraph<u64, u64, u64> {
    /// Induces a subgraph from the original graph based on a list of vertices.
    ///
    /// This function creates a new subgraph containing only the vertices specified in
    /// the input list and the edges between them from the original graph.
    ///
    /// # Arguments
    ///
    /// * `vertex_list` - A vector of vertex IDs to include in the subgraph
    ///
    /// # Returns
    ///
    /// A `CSRSubGraph` structure representing the induced subgraph in CSR (Compressed Sparse Row) format
    ///
    /// # Implementation Details
    ///
    /// 1. Creates a HashSet from the input vertex list for efficient lookups
    /// 2. Initializes data structures to store the new graph representation:
    ///    - vertex_list: Stores each vertex and its offset in the neighbor list
    ///    - neighbor_list: Stores all neighbors in a flattened array
    ///    - vertex_index: Maps original vertex IDs to their positions in the new graph
    /// 3. For each vertex in the input set:
    ///    - Filters its neighbors to include only those present in the input set
    ///    - Adds the vertex to the vertex list with its current offset
    ///    - Updates the vertex index with the vertex's position
    ///    - Appends the filtered neighbors to the neighbor list
    ///
    /// The resulting subgraph maintains the same connectivity as the original graph
    /// but contains only the specified vertices and the edges between them.
    pub fn induce_subgraph(&self, vertex_list: &Vec<u64>) -> CSRSubGraph<u64, u64, u64> {
        // Create a set from the vertex list for efficient membership testing
        let vertex_set: HashSet<u64> = HashSet::from_iter(vertex_list.iter().cloned());
        let vertex_count = vertex_set.len() as u64;

        // Initialize data structures for the subgraph
        let mut vertex_list = Vec::<(u64, u64)>::new();
        let mut neighbor_list = Vec::<u64>::new();
        let mut vertex_index = HashMap::<u64, usize>::new();
        let mut current_offset = 0u64;

        // Process each vertex to build the subgraph structure
        for (vertex_list_id, vertex) in vertex_set.iter().enumerate() {
            // Get neighbors of the current vertex that are also in the vertex set
            let mut neighbors = self.read_neighbor(vertex).into_iter().filter(|n| {
                vertex_set.contains(n)
            }).collect::<Vec<_>>();

            // Add vertex with its offset in the neighbor list
            vertex_list.push((*vertex, current_offset));

            // Map the original vertex ID to its position in the new subgraph
            vertex_index.insert(*vertex, vertex_list_id);

            // Update the current offset for the next vertex
            current_offset += neighbors.len() as u64;

            // Add the filtered neighbors to the neighbor list
            neighbor_list.append(&mut neighbors);
        }

        // Construct and return the subgraph
        CSRSubGraph {
            vertex_count,
            vertex_list,
            neighbor_list,
            vertex_index,
        }
    }

    /// Creates a CSR subgraph that contains all vertices from the original graph.
    ///
    /// This function builds a subgraph that is effectively a copy of the original graph
    /// but in CSRSubGraph format. Unlike selective sub-graphs, this includes all vertices.
    ///
    /// # Returns
    /// * `CSRSubGraph<u64, u64, u64>` - A subgraph containing all vertices from the original graph
    pub fn induce_graph(&self) -> CSRSubGraph<u64, u64, u64> {
        // Get the total number of vertices in the original graph
        let vertex_count = self.vertex_count;

        // Create the vertex list by pairing each vertex ID with its offset in the neighbor list
        // This builds the foundation of the CSR representation
        let vertex_list = self.vertex_list().iter().cloned().zip(
            self.offsets.iter().cloned()
        ).collect::<Vec<(_, _)>>();

        // Copy the neighbor list directly from the original graph
        // This preserves all edge connections
        let neighbor_list = self.neighbor_list.clone();

        // Build a lookup map from vertex IDs to their positions in the vertex list
        // This enables efficient vertex lookups during graph operations
        let mut vertex_index = HashMap::<u64, usize>::new();
        for (vertex_list_id, (vertex_id, _)) in vertex_list.iter().enumerate() {
            vertex_index.insert(*vertex_id, vertex_list_id);
        }

        // Construct and return the complete subgraph with all components
        CSRSubGraph {
            vertex_count,   // Total number of vertices
            vertex_list,    // List of vertex ID and offset pairs
            neighbor_list,  // List of all neighboring vertices (edges)
            vertex_index    // Lookup map for efficient vertex access
        }
    }
}

/// Module for testing vertex encoding and decoding functionality.
#[cfg(test)]
pub mod test_type {
    use super::*;

    /// Tests encoding and decoding of Vertex<u32>.
    /// This test ensures that a Vertex with an u32 identifier can be properly
    /// serialized and deserialized while maintaining data integrity.
    #[test]
    fn test_vertex_u32_encode_decode() {
        // Create a test vertex with u32 vertex_id
        let vertex = Vertex {
            vertex_id: 42u32,
            timestamp: 1234567890,
            tomb: 0,
        };

        // Encode the vertex to its binary representation
        let encoded = vertex.encode_topology();

        // Verify the encoded binary has the expected size (4 bytes for u32 + 8 bytes for timestamp + 1 byte for tomb)
        assert_eq!(encoded.len(), 13);

        // Decode the binary back to a Vertex
        let decoded = Vertex::<u32>::from_bytes_topology(&encoded).unwrap();

        // Ensure all fields match the original values
        assert_eq!(decoded.vertex_id, vertex.vertex_id);
        assert_eq!(decoded.timestamp, vertex.timestamp);
        assert_eq!(decoded.tomb, vertex.tomb);
    }

    /// Tests encoding and decoding of Vertex<u64>.
    /// This test verifies that a Vertex with an u64 identifier can be correctly
    /// serialized and deserialized while preserving all field values.
    #[test]
    fn test_vertex_u64_encode_decode() {
        // Create a test vertex with u64 vertex_id
        let vertex = Vertex {
            vertex_id: 9876543210u64,
            timestamp: 1234567890,
            tomb: 0,
        };

        // Encode the vertex to its binary representation
        let encoded = vertex.encode_topology();

        // Verify the encoded binary has the expected size (8 bytes for u64 + 8 bytes for timestamp + 1 byte for tomb)
        assert_eq!(encoded.len(), 17);

        // Decode the binary back to a Vertex
        let decoded = Vertex::<u64>::from_bytes_topology(&encoded).unwrap();

        // Ensure all fields match the original values
        assert_eq!(decoded.vertex_id, vertex.vertex_id);
        assert_eq!(decoded.timestamp, vertex.timestamp);
        assert_eq!(decoded.tomb, vertex.tomb);
    }

    /// Tests error handling during decoding with invalid input.
    /// This test ensures that the decoder properly handles malformed or insufficient data.
    #[test]
    fn test_invalid_decode() {
        // Test with obviously insufficient bytes
        let invalid_bytes = vec![1, 2, 3];
        let result = Vertex::<u32>::from_bytes_topology(&invalid_bytes);
        assert!(result.is_none());

        // Test with almost valid bytes (missing the tomb byte)
        let almost_valid_bytes = vec![
            1, 0, 0, 0,                 // vertex_id (u32)
            10, 0, 0, 0, 0, 0, 0, 0,    // timestamp (u64)
            // Missing tomb byte
        ];
        let result = Vertex::<u32>::from_bytes_topology(&almost_valid_bytes);
        assert!(result.is_none());
    }

    /// Tests encoding and decoding with different tomb values.
    /// This test verifies that different tomb states are correctly preserved
    /// through the serialization and deserialization process.
    #[test]
    fn test_tomb_values() {
        // Test with multiple possible tomb values
        for tomb_value in [0, 1, 2] {
            let vertex = Vertex {
                vertex_id: 1u32,
                timestamp: 1234567890,
                tomb: tomb_value,
            };

            // Encode and then decode the vertex
            let encoded = vertex.encode_topology();
            let decoded = Vertex::<u32>::from_bytes_topology(&encoded).unwrap();

            // Verify the tomb value is preserved
            assert_eq!(decoded.tomb, tomb_value);
        }
    }

    /// Tests the file loading functionality of CSRGraph
    ///
    /// Verifies that a graph can be correctly loaded from a file by checking:
    /// - The expected vertex count
    /// - The correct size of the offsets array
    /// - The correct size of the neighbor list
    ///
    /// Uses the "oregon.graph" dataset as test data
    #[test]
    fn test_read_from_file() {
        let csr_graph = CSRGraph::from_graph_file("data/example.graph");
        assert_eq!(13, csr_graph.vertex_count);
        assert_eq!(13, csr_graph.offsets.len());
        assert_eq!(20, csr_graph.neighbor_list.len());
    }

    /// Tests the neighbor retrieval functionality of CSRGraph
    ///
    /// This comprehensive test:
    /// 1. Loads a small example graph
    /// 2. Prints debug information about the loaded graph structure
    /// 3. Verifies neighbor lists for multiple vertices against expected values
    /// 4. Tests edge cases like vertices with no neighbors
    /// 5. Tests handling of non-existent vertices
    ///
    /// The test uses a predefined set of expected neighbor relationships
    /// to validate the correctness of the CSR graph implementation.
    #[test]
    fn test_read_neighbor() {
        // Load the test graph from file
        let csr_graph = CSRGraph::<u64, u64, u64>::from_graph_file("data/example.graph");

        // Print graph structure for debugging
        println!("Vertex Count: {}", csr_graph.vertex_count);
        println!("Offsets: {:?}", csr_graph.offsets);
        println!("Neighbor List: {:?}", csr_graph.neighbor_list);

        // Define ground truth - expected neighbors for each vertex
        let ground_truth: Vec<(u64, Vec<u64>)> = vec![
            (0, vec![2]),
            (1, vec![0, 2, 3]),
            (2, vec![3]),
            (3, vec![0, 4, 11]),
            (4, vec![6, 7]),
            (5, vec![4]),
            (6, vec![5]),
            (7, vec![3, 8, 9]),
            (8, vec![9, 10]),
            (10, vec![7, 9]),
            (11, vec![12]),
            (12, vec![])
        ];

        // Test each vertex against its expected neighbor list
        for (vertex, expected_neighbors) in ground_truth {
            let neighbors = csr_graph.read_neighbor(&vertex);
            println!("Vertex {}: Expected: {:?}, Actual: {:?}",
                     vertex, expected_neighbors, neighbors);

            assert_eq!(neighbors, expected_neighbors,
                       "Neighbors of vertex {} don't match expected values", vertex);
        }

        // Test behavior with non-existent vertex
        let non_existent_vertex = 13;
        let neighbors = csr_graph.read_neighbor(&non_existent_vertex);
        assert_eq!(neighbors, vec![],
                   "Non-existent vertex should return empty neighbor list");

        println!("All tests passed!");
    }

    /// Tests the vertex_list functionality of CSRGraph
    ///
    /// This test verifies that:
    /// 1. The vertex_list method returns all vertices in the graph
    /// 2. The returned list has the correct length
    /// 3. All expected vertices are present in the list
    /// 4. The vertices are returned in the expected order (sequential from 0)
    ///
    /// Uses the "example.graph" dataset as test data
    #[test]
    fn test_vertex_list() {
        // Load the test graph from file
        let csr_graph =
            CSRGraph::<u64, u64, u64>::from_graph_file("data/example.graph");
        let loaded_vertex_list = csr_graph.vertex_list();

        // Print the loaded vertex list for debugging
        println!("Loaded vertex list: {:?}", loaded_vertex_list);

        // Verify the vertex count matches the loaded list length
        assert_eq!(csr_graph.vertex_count as usize, loaded_vertex_list.len(),
                   "Vertex list length should match vertex_count");

        // Based on previous tests, we know example.graph should have vertices 0-12
        // Create expected vertex list (sequential from 0 to vertex_count-1)
        let expected_vertex_list: Vec<u64> = (0..csr_graph.vertex_count).collect();

        // Verify that the loaded list matches the expected list
        assert_eq!(expected_vertex_list, loaded_vertex_list,
                   "Vertex list should contain all vertices from 0 to vertex_count-1");

        // Test specific vertices that we know should exist based on previous tests
        for vertex_id in &[0u64, 1u64, 3u64, 7u64, 12u64] {
            assert!(loaded_vertex_list.contains(vertex_id),
                    "Vertex list should contain vertex {}", vertex_id);
        }

        // Test that vertices beyond vertex_count are not included
        assert!(!loaded_vertex_list.contains(&13u64),
                "Vertex list should not contain non-existent vertices");

        println!("All vertex_list tests passed!");
    }

    /// Tests the has_vertex functionality of CSRGraph
    ///
    /// This test verifies that:
    /// 1. The has_vertex method correctly identifies existing vertices
    /// 2. The has_vertex method correctly identifies non-existing vertices
    /// 3. The has_vertex method handles edge cases properly
    ///
    /// Uses the "example.graph" dataset as test data
    #[test]
    fn test_vertex_exist() {
        // Load the test graph from file
        let csr_graph = CSRGraph::<u64, u64, u64>::from_graph_file("data/example.graph");

        println!("Testing has_vertex functionality");

        // Test vertices that should exist (based on previous tests)
        for vertex_id in 0..csr_graph.vertex_count {
            assert!(csr_graph.has_vertex(&vertex_id),
                    "Vertex {} should exist in the graph", vertex_id);
        }

        // Test specific vertices that we know should exist based on previous tests
        for &vertex_id in &[0u64, 1u64, 3u64, 7u64, 12u64] {
            assert!(csr_graph.has_vertex(&vertex_id),
                    "Vertex {} should exist in the graph", vertex_id);
        }

        // Test vertices that should not exist
        let non_existent_vertices = [
            csr_graph.vertex_count,      // First vertex beyond range
            csr_graph.vertex_count + 1,  // Another vertex beyond range
            u64::MAX                     // Edge case: maximum possible vertex id
        ];

        for &vertex_id in &non_existent_vertices {
            assert!(!csr_graph.has_vertex(&vertex_id),
                    "Vertex {} should not exist in the graph", vertex_id);
        }

        println!("All has_vertex tests passed!");
    }

    /// Tests the has_edge functionality of CSRGraph
    ///
    /// This test verifies that:
    /// 1. The has_edge method correctly identifies existing edges
    /// 2. The has_edge method correctly identifies non-existing edges
    /// 3. The has_edge method handles edge cases properly
    ///
    /// Uses the "example.graph" dataset as test data
    #[test]
    fn test_edge_exist() {
        // Load the test graph from file
        let csr_graph = CSRGraph::<u64, u64, u64>::from_graph_file("data/example.graph");

        println!("Testing has_edge functionality");

        // Test edges that should exist based on the ground truth from test_read_neighbor
        let existing_edges = [
            (0, 2),  // Vertex 0 has neighbor 2
            (1, 0),  // Vertex 1 has neighbors 0, 2, 3
            (1, 2),
            (1, 3),
            (2, 3),  // Vertex 2 has neighbor 3
            (3, 0),  // Vertex 3 has neighbors 0, 4, 11
            (3, 4),
            (3, 11),
            (4, 6),  // Vertex 4 has neighbors 6, 7
            (4, 7),
            (7, 9),  // Vertex 7 has neighbors 3, 8, 9
            (11, 12) // Vertex 11 has neighbor 12
        ];

        for &(src, dst) in &existing_edges {
            assert!(csr_graph.has_edge(&src, &dst),
                    "Edge ({} -> {}) should exist in the graph", src, dst);
        }

        // Test edges that should not exist
        let non_existing_edges = [
            (0, 1),   // Vertex 0 doesn't have neighbor 1
            (2, 0),   // Vertex 2 doesn't have neighbor 0
            (12, 11), // Vertex 12 has no neighbors
            (3, 3),   // Self-loops are not in the example graph
            (7, 0),   // Directed edge that doesn't exist
            (csr_graph.vertex_count, 0), // Invalid source vertex
            (0, csr_graph.vertex_count)  // Invalid destination vertex
        ];

        for &(src, dst) in &non_existing_edges {
            assert!(!csr_graph.has_edge(&src, &dst),
                    "Edge ({} -> {}) should not exist in the graph", src, dst);
        }

        // Test edges with vertices that don't exist in the graph
        let out_of_range_vertex = csr_graph.vertex_count + 5;
        assert!(!csr_graph.has_edge(&out_of_range_vertex, &0),
                "Edge with non-existent source vertex should not exist");
        assert!(!csr_graph.has_edge(&0, &out_of_range_vertex),
                "Edge with non-existent destination vertex should not exist");

        println!("All has_edge tests passed!");
    }

    /// Tests the all() method of CSRGraph which returns the complete graph structure
    ///
    /// This test verifies that:
    /// 1. The all() method returns a map containing all vertices in the graph
    /// 2. Each vertex is correctly mapped to its list of neighbors
    /// 3. The data structure matches the expected format
    ///
    /// Uses the "example.graph" dataset as test data
    #[test]
    fn test_all() {
        // Load the test graph from file
        let csr_graph = CSRGraph::<u64, u64, u64>::from_graph_file("data/example.graph");

        println!("Testing all() method");

        // Get the complete graph representation
        let all_graph_data = csr_graph.all();

        // Verify the map contains all vertices
        assert_eq!(csr_graph.vertex_count as usize, all_graph_data.len(),
                   "The map should contain all vertices");

        // Define ground truth - expected neighbors for each vertex, based on test_read_neighbor
        let ground_truth: Vec<(u64, Vec<u64>)> = vec![
            (0, vec![2]),
            (1, vec![0, 2, 3]),
            (2, vec![3]),
            (3, vec![0, 4, 11]),
            (4, vec![6, 7]),
            (5, vec![4]),
            (6, vec![5]),
            (7, vec![3, 8, 9]),
            (8, vec![9, 10]),
            (10, vec![7, 9]),
            (11, vec![12]),
            (12, vec![])
        ];

        // Verify each vertex has the correct neighbors
        for (vertex, expected_neighbors) in &ground_truth {
            // Check that the vertex exists in the map
            assert!(all_graph_data.contains_key(&vertex),
                    "Map should contain vertex {}", vertex);

            // Get the entry for this vertex
            if let Some((returned_vertex, neighbors)) = all_graph_data.get(&vertex) {
                // Verify the returned vertex ID matches the key
                assert_eq!(*returned_vertex, *vertex,
                           "The returned vertex ID should match the key");

                // Verify the neighbors match the expected list
                assert_eq!(neighbors, expected_neighbors,
                           "Neighbors of vertex {} don't match expected values", vertex);
            }
        }

        // Test that vertices not in ground truth also have correct data
        // (For vertices in the graph but not specifically listed in ground truth)
        for vertex_id in 0..csr_graph.vertex_count {
            if !ground_truth.iter().any(|(v, _)| *v == vertex_id) {
                // If this vertex isn't in our ground truth list
                if let Some((returned_vertex, neighbors)) = all_graph_data.get(&vertex_id) {
                    // Verify the returned vertex ID matches the key
                    assert_eq!(*returned_vertex, vertex_id,
                               "The returned vertex ID should match the key");

                    // Verify neighbors match what read_neighbor would return
                    let expected_neighbors = csr_graph.read_neighbor(&vertex_id);
                    assert_eq!(neighbors, &expected_neighbors,
                               "Neighbors should match read_neighbor result");
                }
            }
        }

        println!("All all() method tests passed!");
    }

    /// Tests the read_neighbor functionality of the CSRSubGraph implementation.
    ///
    /// This test verifies that the induced subgraph correctly maintains connectivity
    /// between vertices from the original graph.
    ///
    /// # Test Procedure
    ///
    /// 1. Loads a test graph from a file
    /// 2. Creates a subgraph containing only vertices [0, 1, 2, 3]
    /// 3. Prints debug information about the subgraph structure
    /// 4. Defines expected neighbor relationships as ground truth data
    /// 5. For each vertex in the ground truth:
    ///    - Retrieves its neighbors from the subgraph
    ///    - Compares the actual neighbors with the expected ones
    ///    - Fails the test if they don't match
    /// 6. Tests behavior with a non-existent vertex (should return empty list)
    ///
    /// # Expected Results
    ///
    /// The test expects specific neighbor relationships for each vertex:
    /// - Vertex 0 should have neighbor: [2]
    /// - Vertex 1 should have neighbors: [0, 2, 3]
    /// - Vertex 2 should have neighbor: [3]
    /// - Vertex 3 should have neighbor: [0]
    /// - A non-existent vertex (4) should return an empty neighbor list
    #[test]
    fn test_subgraph_read_neighbor() {
        // Load the test graph from file
        let csr_graph = CSRGraph::<u64, u64, u64>::from_graph_file("data/example.graph");
        let csr_subgraph = csr_graph.induce_subgraph(&vec![0, 1, 2, 3]);

        // Print graph structure for debugging
        println!("Vertex Count: {}", csr_subgraph.vertex_count);
        println!("Offsets: {:?}", csr_subgraph.vertex_list);
        println!("Neighbor List: {:?}", csr_subgraph.neighbor_list);

        // Define ground truth - expected neighbors for each vertex
        let ground_truth: Vec<(u64, Vec<u64>)> = vec![
            (0, vec![2]),
            (1, vec![0, 2, 3]),
            (2, vec![3]),
            (3, vec![0]),
        ];

        // Test each vertex against its expected neighbor list
        for (vertex, expected_neighbors) in ground_truth {
            let neighbors = csr_subgraph.read_neighbor(&vertex);
            println!("Vertex {}: Expected: {:?}, Actual: {:?}",
                     vertex, expected_neighbors, neighbors);

            assert_eq!(neighbors, expected_neighbors,
                       "Neighbors of vertex {} don't match expected values", vertex);
        }

        // Test behavior with non-existent vertex
        let non_existent_vertex = 4;
        let neighbors = csr_subgraph.read_neighbor(&non_existent_vertex);
        assert_eq!(neighbors, vec![],
                   "Non-existent vertex should return empty neighbor list");

        println!("All tests passed!");
    }

    #[test]
    fn test_subgraph_vertex_list() {
        // Load the test graph from file
        let csr_graph = CSRGraph::<u64, u64, u64>::from_graph_file("data/example.graph");
        let csr_subgraph = csr_graph.induce_subgraph(&vec![0, 1, 2, 3]);

        let expected_vertex_list = csr_subgraph.vertex_list();
        assert_eq!(expected_vertex_list.len(), vec![0, 1, 2, 3].len());
        assert_eq!(expected_vertex_list.iter().cloned().collect::<HashSet<_>>(),
                   vec![0, 1, 2, 3].iter().cloned().collect::<HashSet<_>>());
        println!("All tests passed!");
    }

    /// Tests the vertex existence checking functionality of the CSRSubGraph implementation.
    ///
    /// This test verifies that the subgraph correctly identifies which vertices are
    /// present in it after being induced from the original graph.
    ///
    /// # Test Procedure
    ///
    /// 1. Loads a test graph from a file
    /// 2. Creates a subgraph containing only vertices [0, 1, 2, 3]
    /// 3. Tests that each vertex from the input list exists in the subgraph:
    ///    - Checks vertices 0, 1, 2, 3 (should all return true)
    /// 4. Tests that vertices not in the input list don't exist in the subgraph:
    ///    - Checks vertices 4, 5 (should all return false)
    ///
    /// # Expected Results
    ///
    /// - The `has_vertex` method should return true for vertices 0, 1, 2, and 3
    /// - The `has_vertex` method should return false for vertices 4 and 5
    #[test]
    fn test_subgraph_vertex_exist() {
        // Load the test graph from file
        let csr_graph = CSRGraph::<u64, u64, u64>::from_graph_file("data/example.graph");
        let csr_subgraph = csr_graph.induce_subgraph(&vec![0, 1, 2, 3]);

        // Test that expected vertices exist in the subgraph
        for expected_vertex in vec![0, 1, 2, 3] {
            assert!(csr_subgraph.has_vertex(&(expected_vertex as u64)));
        }

        // Test that vertices not in the input list don't exist in the subgraph
        for unexpected_vertex in vec![4, 5] {
            assert!(!csr_subgraph.has_vertex(&(unexpected_vertex as u64)));
        }
        println!("All tests passed!");
    }

    /// Tests the edge existence checking functionality of the CSRSubGraph implementation.
    ///
    /// This test verifies that the subgraph correctly identifies which edges are
    /// present after being induced from the original graph.
    ///
    /// # Test Procedure
    ///
    /// 1. Loads a test graph from a file
    /// 2. Creates a subgraph containing only vertices [0, 1, 2, 3]
    /// 3. Tests that expected edges exist in the subgraph:
    ///    - Checks for edges: (0,2), (1,0), (1,2), (1,3), (2,3), (3,0)
    ///    - All these should return true when checked with has_edge
    /// 4. Tests that edges not in the original graph or involving vertices
    ///    not in the subgraph don't exist:
    ///    - Checks for edges: (3,1), (4,2)
    ///    - These should return false when checked with has_edge
    ///
    /// # Expected Results
    ///
    /// - The `has_edge` method should return true for all expected edges
    /// - The `has_edge` method should return false for:
    ///   * Edges that didn't exist in the original graph (like 3->1)
    ///   * Edges involving vertices that aren't in the subgraph (like 4->2)
    #[test]
    fn test_subgraph_edge_exist() {
        // Load the test graph from file
        let csr_graph = CSRGraph::<u64, u64, u64>::from_graph_file("data/example.graph");
        let csr_subgraph = csr_graph.induce_subgraph(&vec![0, 1, 2, 3]);

        // Test that expected edges exist in the subgraph
        for expected_edge in vec![
            (0, 2), (1, 0), (1, 2), (1, 3), (2, 3), (3, 0)] {
            assert!(csr_subgraph.has_edge(&(expected_edge.0 as u64), &(expected_edge.1 as u64)));
        }

        // Test that unexpected edges don't exist in the subgraph
        for unexpected_edge in vec![
            (3, 1), (4, 2)] {
            assert!(!csr_subgraph.has_edge(&(unexpected_edge.0 as u64), &(unexpected_edge.1 as u64)));
        }

        println!("All tests passed!");
    }

    /// Tests the comprehensive all() method of the CSRSubGraph implementation.
    ///
    /// This test verifies that the all() method correctly returns a complete
    /// representation of the graph with all vertices and their neighbor relationships.
    ///
    /// # Test Procedure
    ///
    /// 1. Loads a test graph from a file
    /// 2. Creates a subgraph containing only vertices [0, 1, 2, 3]
    /// 3. Calls the all() method to get a complete map representation of the graph
    /// 4. Performs multiple validation checks:
    ///    - Confirms the returned map has the correct number of entries (should match vertex_count)
    ///    - Verifies each expected vertex (0-3) exists in the map
    ///    - For each vertex, checks that:
    ///      * The vertex ID in the map value matches the key
    ///      * The neighbor list matches the expected neighbors
    ///    - For any vertices not explicitly covered in the ground truth data,
    ///      verifies their data is consistent with the read_neighbor method
    ///
    /// # Expected Results
    ///
    /// - The returned map should contain exactly vertex_count entries
    /// - For vertices 0-3, the map should contain:
    ///   * Vertex 0 with neighbors [2]
    ///   * Vertex 1 with neighbors [0, 2, 3]
    ///   * Vertex 2 with neighbors [3]
    ///   * Vertex 3 with neighbors [0]
    /// - For each entry, the vertex ID in the value should match the key
    /// - For all vertices, the neighbor list should match what read_neighbor returns
    #[test]
    fn test_subgraph_all() {
        // Load the test graph from file
        let csr_graph = CSRGraph::<u64, u64, u64>::from_graph_file("data/example.graph");
        let csr_subgraph = csr_graph.induce_subgraph(&vec![0, 1, 2, 3]);
        println!("Testing all() method");

        // Get the complete graph representation
        let all_graph_data = csr_subgraph.all();

        // Verify the map contains all vertices
        assert_eq!(csr_subgraph.vertex_count as usize, all_graph_data.len(),
                   "The map should contain all vertices");

        // Define ground truth - expected neighbors for each vertex, based on test_read_neighbor
        let ground_truth: Vec<(u64, Vec<u64>)> = vec![
            (0, vec![2]),
            (1, vec![0, 2, 3]),
            (2, vec![3]),
            (3, vec![0])
        ];

        // Verify each vertex has the correct neighbors
        for (vertex, expected_neighbors) in &ground_truth {
            // Check that the vertex exists in the map
            assert!(all_graph_data.contains_key(&vertex),
                    "Map should contain vertex {}", vertex);

            // Get the entry for this vertex
            if let Some((returned_vertex, neighbors)) = all_graph_data.get(&vertex) {
                // Verify the returned vertex ID matches the key
                assert_eq!(*returned_vertex, *vertex,
                           "The returned vertex ID should match the key");

                // Verify the neighbors match the expected list
                assert_eq!(neighbors, expected_neighbors,
                           "Neighbors of vertex {} don't match expected values", vertex);
            }
        }

        // Test that vertices not in ground truth also have correct data
        // (For vertices in the graph but not specifically listed in ground truth)
        for vertex_id in 0..csr_graph.vertex_count {
            if !ground_truth.iter().any(|(v, _)| *v == vertex_id) {
                // If this vertex isn't in our ground truth list
                if let Some((returned_vertex, neighbors)) = all_graph_data.get(&vertex_id) {
                    // Verify the returned vertex ID matches the key
                    assert_eq!(*returned_vertex, vertex_id,
                               "The returned vertex ID should match the key");

                    // Verify neighbors match what read_neighbor would return
                    let expected_neighbors = csr_graph.read_neighbor(&vertex_id);
                    assert_eq!(neighbors, &expected_neighbors,
                               "Neighbors should match read_neighbor result");
                }
            }
        }
        println!("All all() method tests passed!");
    }

    /// Tests the induce_graph functionality which creates a full subgraph from the original graph.
    ///
    /// This test ensures that the induce_graph method correctly:
    /// 1. Creates a complete subgraph containing all vertices from the original graph
    /// 2. Preserves the neighbor relationships between vertices
    /// 3. Handles edge cases like non-existent vertices appropriately
    #[test]
    fn test_induce_all() {
        // Load the original graph from the test data file
        let csr_graph_origin =
            CSRGraph::<u64, u64, u64>::from_graph_file("data/example.graph");

        // Create a full subgraph containing all vertices from the original graph
        let csr_graph = csr_graph_origin.induce_graph();

        // Print the subgraph structure for debugging and verification
        println!("Vertex Count: {}", csr_graph.vertex_count);
        println!("Offsets: {:?}", csr_graph.vertex_list);
        println!("Neighbor List: {:?}", csr_graph.neighbor_list);

        // Define the expected neighbor relationships for each vertex
        // Each tuple contains (vertex_id, [list of expected neighbor vertices])
        let ground_truth: Vec<(u64, Vec<u64>)> = vec![
            (0, vec![2]),
            (1, vec![0, 2, 3]),
            (2, vec![3]),
            (3, vec![0, 4, 11]),
            (4, vec![6, 7]),
            (5, vec![4]),
            (6, vec![5]),
            (7, vec![3, 8, 9]),
            (8, vec![9, 10]),
            (10, vec![7, 9]),
            (11, vec![12]),
            (12, vec![])
        ];

        // Verify that each vertex has the expected neighbors in the induced subgraph
        for (vertex, expected_neighbors) in ground_truth {
            let neighbors = csr_graph.read_neighbor(&vertex);
            println!("Vertex {}: Expected: {:?}, Actual: {:?}",
                     vertex, expected_neighbors, neighbors);

            // Assert that the actual neighbors match the expected neighbors
            assert_eq!(neighbors, expected_neighbors,
                       "Neighbors of vertex {} don't match expected values", vertex);
        }

        // Test handling of vertices that don't exist in the graph
        // Should return an empty neighbor list
        let non_existent_vertex = 13;
        let neighbors = csr_graph.read_neighbor(&non_existent_vertex);
        assert_eq!(neighbors, vec![],
                   "Non-existent vertex should return empty neighbor list");

        println!("All tests passed!");
    }
}