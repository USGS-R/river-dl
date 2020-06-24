

rule prep_data:
    output:
        "{outdir}/ex_{experiment}/prepped.npz"
    params:
        pt_vars='both',
        ft_vars='both',
    shell:
         """
         python prep_data.py -o {output[0]} -p {params.pt_vars} -f {params.ft_vars}
         """


