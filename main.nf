process generate_safetensors { 
  label 'gpu_process'

  input: 
  path sequences // this can be a fasta or a fastq or a fastq.gz

  output:
  path "*_combined.safetensors", emit: combined_safetensor

  script:
  """
  source actvate ${params.evo2_venv}/bin/activate

  """
}


process verify_safetensors { 
  input:
  path combined_safetensor

  output:
  path "safetensor.dimension_check.txt"

  script:
  """
  """
}