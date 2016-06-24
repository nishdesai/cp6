#!/bin/bash

#args: param_file input_dir output_dir log_path

if (( $# != 4 )); then
printf "Usage: %s param_file input_dir output_dir log_path\n" "$0" >&2
exit 1
fi

#load params
#params:
#    niter: number of iterations
#    nlabels: number of labels to run inference for
source $1;

mkdir -p "$3"
mkdir -p /tmp/blog
cp -r code/* /tmp/blog

#generate BLOG files
printf "Generating BLOG files.\n"
python2.7 generate.py $2 /tmp/blog $nlabels

#Start compiling and running blog files
for i in `seq $nlabels`; do
    pushd /tmp/blog

    printf "Compiling BLOG code for label $i.\n"
    ${ENGROOT}/swift -i eval_cp6_$(( $i-1 )).blog -o eval_cp6_$(( $i-1 )).cpp -e GibbsSampler -n $niter
    sed -i "70d" eval_cp6_$(( $i-1 )).cpp #Delete unused variable name declarations.

    printf "Compiling C++ code for label $i.\n"
    g++ -Ofast -std=c++11 eval_cp6_$(( $i-1 )).cpp random/*.cpp -o eval_cp6_$(( $i-1 )) -larmadillo

    printf "Running Inference for label $i.\n"
    ./eval_cp6_$(( $i-1 )) | tee /tmp/blog/out.txt
    popd

    printf "Processing results for label $i.\n"
    python2.7 parse.py /tmp/blog/out.txt /tmp/blog/labels.csv $(( $i-1 )) $3

    printf "\nOutput generated for label $i. See $3 for partial solution.\n"
done

printf "0.5%.0s " {1..24} > $3/threshold_table.txt
