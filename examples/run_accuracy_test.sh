#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DEFAULT_LOCAL_PATH="/data00/dpskv4-flash-quant/"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}"
    exit 1
fi

MODEL_PATH="${1:-$DEFAULT_LOCAL_PATH}"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at ${MODEL_PATH}"
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
    echo "Current python cannot import torch: ${PYTHON_BIN}"
    exit 1
fi

cd "$PROJECT_ROOT" || exit 1

NUM_LAYERS="${LLM_NUM_LAYERS:-43}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PA_MAX_LENGTH="${PA_MAX_LENGTH:-2048}"

PROMPT_LONG='["女娲是中国上古神话中的创世女神，相传她是一位人首蛇身的伟大女神。在远古洪荒时代，天地一片混沌，没有日月星辰，没有山川草木，也没有人类的存在。女娲独自在这片苍茫的大地上生活了无数个春秋，她感到无比的寂寞和孤独。有一天，女娲来到黄河岸边散步，清澈的河水映照出她美丽的身影。她灵机一动，决定按照自己的模样创造一种新的生命。女娲蹲下身来，从河岸边选取了质地细腻的黄土，开始专心致志地捏造泥人。她心灵手巧，每一个小泥人都塑造得栩栩如生，眉眼五官俱全。女娲对着这些小泥人轻轻吹了一口气，奇迹发生了！这些小泥人竟然活了过来，会说话、会走路、会欢笑，成为了真正的人类。女娲欣喜若狂，继续用双手捏造更多的泥人。她日以继夜地工作，希望大地上能布满人类的足迹。然而，女娲渐渐发现，这样一个一个地捏制实在太慢了，大地如此广袤，需要无穷无尽的人类才能充满生机。聪明机智的女娲想出了一个好办法，她从身旁折下一根长长的藤条，然后将它插入黄河岸边的泥潭之中，沾满浓稠的泥浆后，用力向四面八方挥洒。藤条上溅落的无数泥点落地之后，也都变成了活蹦乱跳的小人儿！这些用藤条甩出的人类，与女娲亲手捏造的泥人一样，都是有血有肉、会说话会思考的真正生命。女娲看到大地上的人类越来越多，心中充满了成就感和喜悦。她教会人类如何狩猎、捕鱼、采摘野果，还教会他们使用火和制造简单的工具。在女娲的庇护下，人类逐渐学会了在自然界中生存繁衍的技能。女娲不仅是人类的创造者，更被后世尊为大地之母和婚姻女神。人们相信，是女娲建立了最初的婚姻制度，让男女结合组建家庭生育后代，从而使人类能够生生不息、代代相传。为了让人类能够长久地生存下去，女娲还亲自巡视大地，观察人类的生活状况，解决他们遇到的困难。她的慈爱和智慧深深地印刻在人类的心中，成为了中华民族最尊敬的远古神明之一。然而，好景不长。在很久很久以后，天地突然发生了一场毁灭性的大灾难。水神共工与火神祝融这两位上古天神因为争夺天帝之位而爆发了激烈的战争。共工是水神的化身，能够呼风唤雨；祝融是火神的代表，能够驱雷掣电。两神在九天之上激战了三天三夜，战斗异常惨烈。天空中电闪雷鸣、狂风暴雨，大地上洪水泛滥、山崩地裂。最终，火神祝融凭借着更强大的神力战胜了共工。共工战败后羞愤交加，怒不可遏，他一气之下撞向了撑天的巨柱不周山。只听得一声惊天动地的巨响，不周山这座连接天地的巨大支柱轰然倒塌。天穹失去了支撑，顿时出现了巨大的裂缝。天空向一侧倾斜，日月星辰开始西移；大地也发生了剧烈的震动，向东倾斜。天空中的裂缝越来越大，滔滔的天河之水从裂缝中倾泻而下，淹没田野村庄，吞噬人间生灵。洪水、火灾、风暴、瘟疫接踵而至，无数百姓在灾难中失去了生命，整个世界陷入了前所未有的浩劫之中。女娲看到自己亲手创造的人类遭受如此惨烈的劫难，心如刀割。她不能眼睁睁地看着人类就此灭亡，于是下定决心要修补破损的天空，拯救她心爱的孩子们。女娲开始了艰苦卓绝的补天工作。她首先攀登到昆仑神山之巅，那里终年积雪、人迹罕至，是采集五色神石的绝佳之地。五色石分别是青色、赤色、黄色、白色和黑色的神奇矿石，蕴含着天地间最纯净的能量。女娲在昆仑山上不辞辛劳地寻找和采集这些珍贵的矿石，无论严寒酷暑，无论风霜雨雪，她从未间断。收集到足够的五色石后，女娲又在山顶燃起了一团熊熊烈火。这团神火燃烧了整整七七四十九天，火焰高达万丈，将整个昆仑山照得如同白昼。女娲将五色神石投入神火之中，用她无比强大的神力将矿石炼化成熔融的岩浆。这些闪烁着五彩光芒的熔浆蕴含着无穷的能量，是修补天穹裂缝的完美材料。女娲用双手捧起五色熔浆，一点点地填补天空上的巨大裂缝。她的动作轻柔而坚定，就像母亲为孩子缝补衣裳一样细致认真。经过了九九八十一天的艰苦努力，女娲终于用五色熔浆将天空的裂缝全部修补完成。天空重新恢复了完整和稳固，不再有天河之水倾泻而下。为了彻底解决天塌地陷带来的连锁灾难，女娲又斩杀了一只作恶的黑龙，用龙的巨大身躯和龙鳞作为材料，进一步加固了天空的薄弱之处。她还砍下神龟的巨足，用作新的天柱，分别竖立在东西南北四个方向，稳稳地支撑起苍茫的天穹。对于四处泛滥的滔天洪水，女娲想出了一个绝妙的方法。她收集了大量的芦苇，将它们烧成灰烬，然后把这些芦苇灰填塞进四处决堤的河流和湖海之中。奇迹再次发生，芦苇灰竟然成功地堵塞了所有的洪水缺口，大地重新恢复了平静和安宁。经历了这场浩劫的人类，在女娲的拯救下终于得以幸存。大地上重新出现了炊烟袅袅的村落，出现了欢声笑语的家庭，出现了男耕女织的祥和景象。人们纷纷走出避难的洞穴和山洞，重新开始重建家园，过上了安居乐业的幸福生活。女娲补天之后，她的伟大功绩被千秋万代传颂不衰。她被后世尊崇为人类的守护神和救世英雄，成为了中华民族共同的母亲和始祖。人们感激她的救命之恩，永远铭记她的慈爱和牺牲精神。为了表达对女娲的敬仰和感恩之情，后世的人们在各地建立女娲庙和女娲祠，供奉她的神像。每年重要的节日和祭祀日，人们都会举行盛大的祭祀活动，献上丰盛的祭品，虔诚地向女娲祈祷，祈求她的保佑和庇护。这些祭祀活动代代相传，延续至今，已经成为了中华民族传统文化的重要组成部分。女娲神话不仅展现了远古先民对世界起源的美好想象，更深刻地体现了中华民族尊崇母性、敬畏自然、团结抗争的民族精神。这个流传了几千年的神话故事，承载着丰富的文化内涵和深刻的历史意义，激励着一代又一代中华儿女团结奋进、自强不息。请简要总结一下这个神话故事的主要内容。"]'
PROMPT_SHORT='["请用128字以内介绍一下量子计算的核心原理。"]'

run_inference() {
    local graph_mode=$1
    local bs=$2
    local prompt=$3
    local desc=$4
    local ep_size=${5:-8}

    export MOJO_GRAPH_MODE="${graph_mode}"
    export MOJO_BACKEND="ascendc"
    export MOJO_PROF="0"
    export MOJO_DISABLE_ASSERTION_REWRITE="${MOJO_DISABLE_ASSERTION_REWRITE:-1}"

    echo ""
    echo "============================================================"
    echo "Graph Mode: ${graph_mode}"
    echo "Test: ${desc}"
    echo "Batch Size: ${bs}, EP_SIZE: ${ep_size}"
    echo "============================================================"

    if [ "$ep_size" -eq 1 ]; then
        ASCEND_RT_VISIBLE_DEVICES=8 \
        "${PYTHON_BIN}" -m examples.llm_inference \
            --model_path "${MODEL_PATH}" \
            --device npu \
            --num_layers "${NUM_LAYERS}" \
            --max_new_tokens "${MAX_NEW_TOKENS}" \
            --pa_max_length "${PA_MAX_LENGTH}" \
            --prompt "${prompt}" \
            --ep_size 1 \
            --batch_size "${bs}"
    else
        export WORLD_SIZE="${ep_size}"
        export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
        export MASTER_PORT="${MASTER_PORT:-6038}"
        export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
        export HCCL_IF_IP="${HCCL_IF_IP:-$(hostname -I | awk '{print $1}')}"
        export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-23456}"
        export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
        export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1200}"
        export RANK_OFFSET="${RANK_OFFSET:-0}"

        PIDS=()
        for((i=8; i<$((${ep_size}+8)); i++))
        do
            export LOCAL_RANK=$(expr $i - 8)
            export RANK_ID=$(expr $i + $RANK_OFFSET)
            export NPU_DEVICE_IDX=$i

            echo "Launching rank ${RANK_ID} (local_rank=${LOCAL_RANK}, npu_device=${i})..."

            "${PYTHON_BIN}" -m examples.llm_inference \
                --model_path "${MODEL_PATH}" \
                --device npu \
                --num_layers "${NUM_LAYERS}" \
                --max_new_tokens "${MAX_NEW_TOKENS}" \
                --pa_max_length "${PA_MAX_LENGTH}" \
                --prompt "${prompt}" \
                --ep_size "${ep_size}" \
                --batch_size "${bs}" &

            PIDS+=($!)
        done

        for pid in "${PIDS[@]}"; do
            wait "$pid" || echo "Process $pid exited with error"
        done
    fi

    echo "============================================================"
    echo "Test '${desc}' [${graph_mode}] finished"
    echo "============================================================"
}

echo ""
echo "############################################################"
echo "#  AscendC Inference Accuracy Test                         #"
echo "#  Comparing eager vs npugraph_ex                          #"
echo "############################################################"

PROMPT_BS2="[\"请用128字以内介绍一下量子计算的核心原理。\", \"女娲是中国上古神话中的创世女神，相传她是一位人首蛇身的伟大女神。在远古洪荒时代，天地一片混沌，没有日月星辰，没有山川草木，也没有人类的存在。女娲独自在这片苍茫的大地上生活了无数个春秋，她感到无比的寂寞和孤独。有一天，女娲来到黄河岸边散步，清澈的河水映照出她美丽的身影。她灵机一动，决定按照自己的模样创造一种新的生命。女娲蹲下身来，从河岸边选取了质地细腻的黄土，开始专心致志地捏造泥人。她心灵手巧，每一个小泥人都塑造得栩栩如生，眉眼五官俱全。女娲对着这些小泥人轻轻吹了一口气，奇迹发生了！这些小泥人竟然活了过来，会说话、会走路、会欢笑，成为了真正的人类。女娲欣喜若狂，继续用双手捏造更多的泥人。她日以继夜地工作，希望大地上能布满人类的足迹。女娲不仅是人类的创造者，更被后世尊为大地之母和婚姻女神。\"]"

for mode in eager npugraph_ex; do
    echo ""
    echo "################################################################"
    echo "#  MODE: ${mode}                                                "
    echo "################################################################"

    run_inference "${mode}" 1 "${PROMPT_LONG}" "BS=1, Long Prompt, EP=8"
    run_inference "${mode}" 1 "${PROMPT_SHORT}" "BS=1, Short Prompt, EP=8"
    run_inference "${mode}" 2 "${PROMPT_BS2}" "BS=2, Short+Long Prompt, EP=8"
done

echo ""
echo "============================================================"
echo "All accuracy tests completed!"
echo "============================================================"
