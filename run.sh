# python generate_v3.py

python evaluate_simulated_conversations.py --model gpt-4o &
python evaluate_simulated_conversations.py --model "gpt-4o-mini" &
python evaluate_simulated_conversations.py --model "gpt-5" &
python evaluate_simulated_conversations.py --model "gpt-5-mini" &
python evaluate_simulated_conversations.py --model "o3" &
python evaluate_simulated_conversations.py --model "o4-mini" &

wait

python evaluate_simulated_conversations_pairwise.py.py --model gpt-4o &
python evaluate_simulated_conversations_pairwise.py.py --model "gpt-4o-mini" &
python evaluate_simulated_conversations_pairwise.py.py --model "gpt-5" &
python evaluate_simulated_conversations_pairwise.py.py --model "gpt-5-mini" &
python evaluate_simulated_conversations_pairwise.py.py --model "o3" &
python evaluate_simulated_conversations_pairwise.py.py --model "o4-mini" &

wait