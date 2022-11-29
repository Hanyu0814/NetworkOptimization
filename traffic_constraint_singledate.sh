#!/bin/bash
#SBATCH --time=47:59:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hyu376@usc.edu
#SBATCH --mem=64G
#SBATCH --account=xiangren_818

module load gcc
module load python/3.7

start_m=$1
start_d=$2
end_m=$3
end_d=$4
year=$5
flag_weekday=$6

out_put_file="Output.out"

echo $start_m , $start_d , $end_m , $end_d

day_start="$year"+$start_m+$start_d
day_end="$year"+$end_m+$end_d

echo $day_start , $day_end

now=`date +"%m_%d_%Y"`
filename=${now}_NumericalResult
mkdir -p "Numerical_result/$filename"
cd "Numerical_result/$filename"

filename2="OD_${start_m}_${start_d}_${end_m}_${end_d}_${year}_netwrokL"
mkdir -p "$filename2"
cd $filename2
cp ../../../Traffic_constraint_singledate_v1.py .
cp ../../../base.py .
cp ../../../pfe_sym.py .
mkdir -p "R_matrix"
mkdir -p "P_matrix"
mkdir -p "X_vector"
mkdir -p "Q_vector"
mkdir -p "Constraints"

python3 Traffic_constraint_singledate_v1.py "${start_m}" "${start_d}" "${end_m}" "${end_d}" "${year}" "${flag_weekday}"


# python3 Traffic.py $num_line $flag_week_day > out_put_file
