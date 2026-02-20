process generate_safetensors { 
  label 'gpu_process'
  
  input: 
  path sequences

  output:
  path "*_chunk.safetensors", emit: chunk_safetensors

  script:
  """
  easyevo2 embed ${sequences} --output temp_embed
  
  # Merge the tiny 100-seq shards easyevo2 creates for this specific chunk
  python ${projectDir}/bin/merge_safetensors.py \\
  --prefix "temp_embed" \\
  --output "${sequences.baseName}_chunk.safetensors" \\
  --cleanup
  """
}

process final_merge {
  label 'cpu_process'
  publishDir 'results/safetensors', mode: 'symlink'

  input:
  val base_name
  path all_chunks // Receives a list of all chunked safetensors

  output:
  path "${base_name}_combined.safetensors", emit: final_safetensor

  script:
  """
  # Merge all the chunks into your final file
  python ${projectDir}/bin/merge_safetensors.py \\
  --files ${all_chunks} \\
  --output "${base_name}_combined.safetensors"
  """
}

process verify_safetensors { 
  label 'cpu_process'
  publishDir 'results/embeddings_summary', mode: 'copy'

  input:
  path combined_safetensor

  output:
  path "safetensor.dimension_check.txt"

  script:
  """
  python ${projectDir}/bin/check_embeddings.py --safetensor ${combined_safetensor} > safetensor.dimension_check.txt
  """
}

workflow {
  // Grab the base name to use for the final output file
  base_name = file(params.sequence).baseName

  // Split into chunks of 50,000 for parallel GPU processing
  seq_chunks = Channel.fromPath(params.sequence)
                      .splitFasta(by: 50000, file: true)

  gen_out = generate_safetensors(seq_chunks)
  
  // .collect() waits for all chunks to finish, then gathers them into a list for final_merge
  merged_out = final_merge(base_name, gen_out.chunk_safetensors.collect())
  
  verify_safetensors(merged_out.final_safetensor)
}
