export RUN_SLOW=1
export RUN_PT_TF_CROSS_TESTS="False"
export RUN_PT_FLAX_CROSS_TESTS="False"
export TRANSFORMERS_TEST_DEVICE="xpu"
export TRANSFORMERS_TEST_DEVICE_SPEC="spec.py"

excel_dir="$1"

echo "+++++++++remove excel dir if exists and create a new++++++++++++"
rm -fr $excel_dir 
mkdir $excel_dir


python -m pytest tests -sv --excelreport="${excel_dir}/all_uts.xlsx" 