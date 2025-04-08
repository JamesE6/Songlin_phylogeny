# Arawan_Phylogeny
Repository accompanying the "An Overview and Bayesian phylogenetic study of the Arawan language family" paper.

The [Trees](/Trees/) folder contains the settings and the output of tree runs conducted.  



## Generating Nexus files

Build the nexus file with the following command; this will use zero for ascertainment bias
when a concept is not observed, and rename/filter doculects and parameters. The experiment
will be called "base" and the output will be written to the "output" directory.

```bash
python .\build_nexus.py -z -l DOCULECT -d .\etc\arawan.txt -p .\etc\concepts.txt .\raw\arawan.20230630.tsv base
```

