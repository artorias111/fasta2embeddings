process generate_safetensors { 
  label 'gpu_process'

  input: 
  path sequences // this can be a fasta or a fastq or a fastq.gz

  output:
  path "*_combined.safetensors", emit: combined_safetensor

  script:
  """
  easyevo2 embed ${sequences} --output ${sequences.baseName}
  """
}


process verify_safetensors { 
  label 'cpu_process'

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
  seq = Channel.fromPath(params.sequence)
  generate_safetensors(seq)
  verify_safetensors(generate_safetensors.out.combined_safetensor)
}