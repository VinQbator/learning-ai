

# How many players in table
NUMBER_OF_SEATS = 2
# Max betsize in simulation environment (shouldn't really matter with discrete relative to pot sizing)
MAX_BET = 100000
# 'norm' (normalized) or 'one-hot' < how to encode player hand ranking from 7642 unique values
RANK_ENCODING = 'norm'

FIRST_RUN_STEPS = 1000
SECOND_RUN_STEPS = 50000
THIRD_RUN_STEPS = 20000
THIRD_RUN_ITERATIONS = 25
THIRD_RUN_WINDOW = 10

def main():
    # Lets start with playing against player that always calls or checks based on which is currently valid move
    # Hopefully this will teach the agent something about hand strength at least
    env = build_environment(ATM(), False)


if __name__ == '__main__':
    main()