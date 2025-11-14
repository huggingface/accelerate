export MASTER_ADDR=localhost
export MASTER_PORT=9998
python -u -m accelerate.commands.launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --config_file sp-alst.accelerate-config.yml \
    sp-alst.py
