import os
import sys
import traceback
import numpy as np

sys.path.append(os.environ["ROOT_PATH"])
from rewards_database import RevolveDatabase, EurekaDatabase
from modules import *
import utils
import random
import prompts
from evolutionary_utils.custom_environment import CustomEnvironment
from typing import Callable, List
import absl.logging as logging
from functools import partial
from utils import *
import hydra
import os
import glob
#from agent.inference import evaluate_policy_n_episodes
from agent.inference import run_main


def load_policy_from_txt(txt_path):
    namespace = {}
    exec(open(txt_path).read(), namespace)
    return namespace['policy']


def check_policy_function_format(code: str) -> bool:
    """
    Accepts either:
    - class Policy with compute_action(self, obs, path)
    - def policy(obs, path)
    """
    # Case 1: class-based
    if re.search(r"class\s+Policy\b", code) and re.search(r"def\s+compute_action\s*\(.*obs.*path.*\)", code):
        return True
    # Case 2: function-based
    if re.search(r"def\s+policy\s*\(.*obs.*path.*\)", code):
        return True
    return False

def generate_valid_policy(policy_generation, in_context_prompt, max_trials: int = 10):
    trials = 0
    while trials < max_trials:
        try:
            policy_code_str = policy_generation.generate_rf(in_context_prompt)

            if not check_policy_function_format(policy_code_str):
                raise ValueError("Invalid policy format")

            # Only return the code string, skip exec
            return policy_code_str, [], None

        except (ValueError, SyntaxError) as e:
            print(f"Policy validation failed: {e}")
            trials += 1
            continue

        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    return None, None, None


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.environ["ROOT_PATH"], "cfg"),
    config_name="generate",
)
def main(cfg):
    print("=== RUNTIME DEBUG ===")
    print("CWD:", os.getcwd())
    print("cfg.database.rewards_dir:", cfg.database.rewards_dir)
    print("ENV ROOT_PATH:", os.environ.get("ROOT_PATH"))
    print("ENV GEN_ID:", os.environ.get("GEN_ID"))
    print("======================")

    #LOAD = int(os.environ.get("LOAD", "0"))       # 0 = generate-only, 1 = evaluate-only
    #EPISODES = int(os.environ.get("EPISODES", "15"))
   # counter_smth=
    env_name = cfg.environment.name

    system_prompt = prompts.types["system_prompt"]
   # env_input_prompt = prompts.types["env_input_prompt"]

    reward_generation = RewardFunctionGeneration(
        system_prompt=system_prompt
    )

    # create log directory
    log_dir = cfg.data_paths.output_logs
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tracker = utils.DataLogger(os.path.join(log_dir, "progress.log"))

    # define a schedule for temperature of sampling
    temp_scheduler = partial(
        utils.linear_decay,
        initial_temp=cfg.database.initial_temp,
        final_temp=cfg.database.final_temp,
        num_iterations=cfg.evolution.num_generations,
    )
    if "revolve" in cfg.evolution.baseline:
        database = partial(
            RevolveDatabase,
            num_islands=cfg.database.num_islands,
            max_size=cfg.database.max_island_size,
            crossover_prob=cfg.database.crossover_prob,
            migration_prob=cfg.database.migration_prob,
            reward_fn_dir=cfg.database.rewards_dir,
            baseline=cfg.evolution.baseline,
        )
    else:
        database = partial(
            EurekaDatabase,
            num_islands=1,  # Eureka for a single island
            max_size=cfg.database.max_island_size,
            reward_fn_dir=cfg.database.rewards_dir,
            baseline=cfg.evolution.baseline,
        )
    current_gen_env = os.environ.get("GEN_ID")
    current_gen_env = int(os.environ.get("GEN_ID", 0))
   # for generation_id in range(0, cfg.evolution.num_generations ):
    for generation_id in range(current_gen_env, current_gen_env+1):
        # fix the temperature for sampling
        temperature = temp_scheduler(iteration=generation_id)
        print(
            f"\n====!!!!!!!!!!!!!===== Generation {generation_id} | Model: {cfg.evolution.baseline} | temperature: {round(temperature, 2)} =========="
        )
        # load all groups if iteration_id > 0, else initialize empty islands
        rewards_database = database(load_islands=not generation_id == 0)
        print("\n=== [DEBUG] Loaded Islands Summary ===")
        for i, island in enumerate(rewards_database._islands):
            print(f"Island {i}: size={island.size}, avg_fit={island.average_fitness_score:.3f}")
        print("======================================\n")
        if generation_id > 0:
            for island in rewards_database._islands:
                island.individuals = [
                    ind for ind in island.individuals
                    if ind.generation_id < generation_id
                ]
        # ✅ verification / debug print
        for island_id, island in enumerate(rewards_database._islands):
            print(f"\nIsland {island_id} contains {len(island.individuals)} individuals:")
            for ind in island.individuals:
                print(f"  gen={ind.generation_id} | fit={ind.fitness_score:.3f} | {ind.fn_file_path}")
                assert ind.generation_id < generation_id, (
                    f"[ERROR] Island {island_id} has individual from current gen "
                    f"{ind.generation_id} >= {generation_id}"
                )
        print("\n=== ON-DISK FITNESS CROSS-CHECK ===")
        for island_id, island in enumerate(rewards_database._islands):
            for ind in island.individuals:
                fit_file = os.path.join(
                    cfg.database.rewards_dir,
                    f"island_{island_id}",
                    "fitness_scores",
                    f"{ind.generation_id}_{ind.counter_id}.txt"
                )
                disk_val = None
                try:
                    with open(fit_file, "r") as f:
                        disk_val = f.read().strip()
                except FileNotFoundError:
                    disk_val = "<MISSING>"
                print(f"Island {island_id} | {ind.generation_id}_{ind.counter_id} | obj.fitness={ind.fitness_score:.6f} | file={fit_file} -> {disk_val}")
        print("====================================\n")


        rew_fn_strings = []  # valid rew fns
        # fitness_scores = []
        island_ids = []
        counter_ids = []
        fitness_scores=[]
        metrics_dicts=[]
        # metrics_dicts = []
        policies = []
        load=0
#       path_debug_policy_file=(r'/home/alkis/Downloads/carla/python_code/Revolve/database/revolve_auto/1/island_2/policies/1_0.txt')

        # for each generation, produce new individuals via mutation or crossover
        counter_env = os.environ.get("COUNTER")
        if counter_env is not None:
            counter_iter = range(int(counter_env), int(counter_env) + 1)
        else:
            counter_iter = range(cfg.evolution.individuals_per_generation)
        print("counter_iter",counter_iter)
        for counter_id in counter_iter:
            if generation_id == 0:  # initially, uniformly populate the islands
                # TODO: to avoid corner cases, populate all islands uniformly
                island_id = random.choice(range(rewards_database.num_islands))
                in_context_samples = (None, None)
                operator_prompt = ""
                logging.info(f"Generation {generation_id}, Counter {counter_id}: island_id={island_id}, type={type(island_id)}")

            else:  # gen_id > 0: start the evolutionary process
                (
                    in_context_samples,
                    island_id,
                    operator,
                ) = rewards_database.sample_in_context(
                    cfg.few_shot, temperature
                )  # weighted sampling of islands and corresponding individuals
                operator = f'{operator}_auto' if 'auto' in cfg.evolution.baseline else operator
                operator_prompt = prompts.types[operator]
            print("island id and coutner it",island_id,counter_id)
            island_ids.append(island_id)
            counter_ids.append(counter_id) 
            # each sample in 'in_context_samples' is a tuple of (fn_path: str, fitness_score: float)
            in_context_prompt = RewardFunctionGeneration.prepare_in_context_prompt(
                in_context_samples,
                operator_prompt,
                evolve=generation_id > 0,
                baseline=cfg.evolution.baseline,
            )
            logging.info(f"Designing reward function for counter {counter_id}")
          #  print("\n=== FINAL PROMPT TO LLM ===\n")
           # print(in_context_prompt)
            #print("\n=== END PROMPT ===\n")
            #generate valid fn str
            if load==0:
                reward_func_str, _, _ = generate_valid_policy(
                    reward_generation, in_context_prompt
                )
            else:
                gen_id = int(os.environ.get("GEN_ID", 1))
                counter_id = int(os.environ.get("COUNTER", 0))
                islands_root = cfg.database.rewards_dir

                # search all islands for the correct file
                found_files = glob.glob(f"{islands_root}/island_*/policies/{gen_id}_{counter_id}.txt")
                if not found_files:
                    raise FileNotFoundError(f"Policy file for gen={gen_id}, counter={counter_id} not found in {islands_root}")
                path_debug_policy_file = found_files[0]
                print(f"[LOAD=1] Using existing counter and generation file: {counter_id}, {gen_id}")
                print(f"[LOAD=1] Using existing policy file: {path_debug_policy_file}")

                with open(path_debug_policy_file, "r") as f:
                    reward_func_str = f.read()

                policy_file = path_debug_policy_file
                # with open(path_debug_policy_file, "r") as f:
                #     reward_func_str = f.read()
                # policy_file=path_debug_policy_file

           # print("reward_func_str",reward_func_str)
            policy_save_dir = os.path.join(cfg.database.rewards_dir, f"island_{island_id}/policies")
            os.makedirs(policy_save_dir, exist_ok=True)
            if load==0:
                policy_file = os.path.join(policy_save_dir, f"{generation_id}_{counter_id}.txt")
                with open(policy_file, "w") as f:
                    f.write(reward_func_str)
            rew_fn_strings.append(reward_func_str)  # should be policy_func_str


            # 4. Evaluate this policy file for N episodes
            print("running inference in main")
           # avg_fitness, fitnesses, avg_metrics = evaluate_policy_n_episodes(policy_file, n_episodes=2)
           # avg_fitness, avg_metrics = 0.01,{}
            results, old_fit = run_main(policy_file,episodes=10)
            avg_fitness = results["fitness"]      # average fitness value
            metrics = results["metrics"]
            print("infeence finnisehd in main")
            fitness_scores.append(avg_fitness)
            metrics_dicts.append(metrics)
            # utils.save_metrics_json(
            #     avg_metrics,
            #     cfg.database.rewards_dir,
            #     island_id,
            #     generation_id,
            #     counter_id,
            # )
            print(" avg_fitness, metrics", avg_fitness, metrics)

            print(f"Individual {counter_id} in island {island_id} | fitness: {avg_fitness}")
            old_dir = os.path.join(cfg.database.rewards_dir, f"island_{island_id}", "old_fit")
            os.makedirs(old_dir, exist_ok=True)

            old_fit_path = os.path.join(old_dir, f"{generation_id}_{counter_id}.txt")
            with open(old_fit_path, "w") as f:
                f.write(f"{old_fit:.4f}\n")
           # policies.append()
          #  json_path = save_policy_summary(policy_file, fitness, fitnesses, metrics)
            #print(f"[policy_history] wrote {json_path}")
       # if len(policies) == 0:
           ## logging.info("No valid reward functions. Hence, no policy trains required.")
            continue

        if generation_id > 0:
            rewards_database.add_individuals_to_islands(
                [generation_id] * len(island_ids),
                counter_ids,
                rew_fn_strings,
                fitness_scores,
                metrics_dicts,
                island_ids,
            )
        else:  # initialization step (generation = 0)
            rewards_database.seed_islands(
                [generation_id] * len(island_ids),
                counter_ids,
                rew_fn_strings,
                fitness_scores,
                metrics_dicts,
                island_ids,
            )

        island_info = [
            {
                island_id: {
                    f"{gen_id}_{count_id}": fitness
                    for gen_id, count_id, fitness in zip(
                        island.generation_ids, island.counter_ids, island.fitness_scores
                    )
                }
            }
            for island_id, island in enumerate(rewards_database._islands)
        ]
        tracker.log({"generation": generation_id, "islands": island_info})


if __name__ == "__main__":
    main()
