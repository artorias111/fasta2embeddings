base_name = file(params.sequence).baseName

process generate_safetensors { 
    label 'gpu_process'
    publishDir "${base_name}.safetensors", mode: 'copy'

    input: 
    path sequences

    output:
    path "*_chunk.safetensors", emit: chunk_safetensors

    script:
    """
    easyevo2 embed ${sequences} \\
        --output "${sequences.baseName}_chunk.safetensors" \\
        --merge
    """
}

process verify_safetensors { 
    label 'cpu_process'
    publishDir "${base_name}.safetensors", mode: 'copy'

    input:
    path all_chunks 

    output:
    path "safetensor.dimension_check.txt"

    script:
    """
    python ${projectDir}/bin/check_embeddings.py --safetensor ${all_chunks} > safetensor.dimension_check.txt
    """
}

workflow {
    seq_chunks = Channel.fromPath(params.sequence)
        .splitFasta(by: params.chunksize, file: true)

    gen_out = generate_safetensors(seq_chunks)
    
    verify_safetensors(gen_out.chunk_safetensors.collect())
}
