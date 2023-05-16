import pandas as pd
import numpy as np
import random
from ortools.linear_solver import pywraplp
from deap import base, creator, tools, algorithms
import functools
from simanneal import Annealer
import time


def read_excel_data(filepath):
    df = pd.read_csv(filepath)
    return df


def process_data(df, n_medical_staff):
    # Convert date and time columns to datetime objects
    df["Surgery Start Date"] = pd.to_datetime(df["Surgery Start Date"])
    df["Surgery End Date"] = pd.to_datetime(df["Surgery End Date"])
    df["Anaesthetic Start"] = pd.to_datetime(df["Anaesthetic Start"])
    df["Surgery Start"] = pd.to_datetime(df["Surgery Start"])
    df["Surgery Finish"] = pd.to_datetime(df["Surgery Finish"])
    df["Anaesthetic Finish"] = pd.to_datetime(df["Anaesthetic Finish"])
    df["Left Theatre"] = pd.to_datetime(df["Left Theatre"])

    # Calculate surgery duration and store it in a new column
    df["Surgery Duration"] = df["Surgery Finish"] - df["Surgery Start"]

    # Extract unique operating rooms and case types
    operating_rooms = df["Theatre"].unique()
    case_types = df["Case Type"].unique()

    # Generate random resource requirements and staff preferences for each case type
    resource_requirements = {
        case_type: np.random.randint(1, 5, size=n_medical_staff)
        for case_type in case_types
    }
    staff_preferences = {
        case_type: np.random.rand(n_medical_staff) * 4 + 1 for case_type in case_types
    }

    return operating_rooms, case_types, resource_requirements, staff_preferences


def create_constraint_programming_model(
    operating_rooms,
    case_types,
    resource_requirements,
    n_medical_staff,
    staff_preferences,
):
    # Create constraint programming model
    solver = pywraplp.Solver.CreateSolver("SCIP")
    x = {}
    for room in operating_rooms:
        for case_type in case_types:
            for staff in range(n_medical_staff):
                x[room, case_type, staff] = solver.BoolVar(
                    f"x_{room}_{case_type}_{staff}"
                )

    # Constraint: Ensure each case type is assigned to a single operating room
    for case_type in case_types:
        solver.Add(
            solver.Sum(
                x[room, case_type, staff]
                for room in operating_rooms
                for staff in range(n_medical_staff)
            )
            == 1
        )

    # Constraint: Ensure each operating room has at most one case type
    for room in operating_rooms:
        solver.Add(
            solver.Sum(
                x[room, case_type, staff]
                for case_type in case_types
                for staff in range(n_medical_staff)
            )
            <= 1
        )

    # Constraint: Ensure resource requirements are met for each case type
    for case_type in case_types:
        for staff in range(n_medical_staff):
            solver.Add(
                solver.Sum(
                    x[room, case_type, staff] * resource_requirements[case_type][staff]
                    for room in operating_rooms
                )
                >= 1
            )

    # Constraint: Ensure that the duration of all surgeries in a room does not exceed the available time
    for room in operating_rooms:
        solver.Add(
            solver.Sum(
                x[room, case_type, staff]
                * df.loc[df["Case Type"] == case_type, "Surgery Duration"].values[0]
                for case_type in case_types
                for staff in range(n_medical_staff)
            )
            <= AVAILABLE_TIME
        )

    # Minimize the sum of staff dissatisfaction
    objective = solver.Objective()
    for room in operating_rooms:
        for case_type in case_types:
            for staff in range(n_medical_staff):
                objective.SetCoefficient(
                    x[room, case_type, staff], staff_preferences[case_type][staff]
                )
    objective.SetMinimization()

    return solver, x


def feasible_individual(individual, model_tuple):
    # Unpack the model tuple
    solver, x, operating_rooms, case_types, n_medical_staff = model_tuple

    # Check the feasibility of an individual using the constraint programming model
    feasible = True

    for i, room in enumerate(operating_rooms):
        for j, case_type in enumerate(case_types):
            for staff in range(n_medical_staff):
                # Assign values from the individual to the decision variables x
                value = float(individual[i][j][staff])
                x[room, case_type, staff].SetLb(value)
                x[room, case_type, staff].SetUb(value)

    # Solve the constraint programming model with the given individual
    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        feasible = False

    # Reset the decision variables x to their original bounds
    for i, room in enumerate(operating_rooms):
        for j, case_type in enumerate(case_types):
            for staff in range(n_medical_staff):
                x[room, case_type, staff].SetLb(0)
                x[room, case_type, staff].SetUb(1)

    return feasible


def evaluation_function(individual, resource_requirements, staff_preferences):
    # Convert the individual to a dictionary for easy processing
    assignment = {
        (i, j, k): individual[i][j][k]
        for i, room in enumerate(operating_rooms)
        for j, case_type in enumerate(case_types)
        for k in range(n_medical_staff)
    }

    # Calculate the fitness of an individual based on the objective function(s)
    total_cost = 0
    total_dissatisfaction = 0

    for (room, case_type, staff), value in assignment.items():
        # Calculate the cost based on the assignment of staff to cases and operating rooms
        total_cost += value * resource_requirements[case_types[case_type]][staff]

        # Calculate the dissatisfaction based on the assignment of staff to cases and operating rooms
        total_dissatisfaction += value * staff_preferences[case_types[case_type]][staff]

    # Return the total cost and dissatisfaction as a tuple
    return total_cost, total_dissatisfaction


class SchedulingProblem(Annealer):
    def __init__(self, state, toolbox):
        super(SchedulingProblem, self).__init__(state)  # important!
        self.toolbox = toolbox

    def move(self):
        """Swaps two staff in a random operating room and case type."""
        # Select random operating room and case type
        room_index = random.randint(0, len(self.state) - 1)
        case_type_index = random.randint(0, len(self.state[room_index]) - 1)

        # Select two random staff to swap
        staff_a_index = random.randint(
            0, len(self.state[room_index][case_type_index]) - 1
        )
        staff_b_index = random.randint(
            0, len(self.state[room_index][case_type_index]) - 1
        )

        # Swap them
        (
            self.state[room_index][case_type_index][staff_a_index],
            self.state[room_index][case_type_index][staff_b_index],
        ) = (
            self.state[room_index][case_type_index][staff_b_index],
            self.state[room_index][case_type_index][staff_a_index],
        )

    def energy(self):
        """Calculates the cost of the current state."""
        return self.toolbox.evaluate(self.state)


def simulated_annealing(toolbox, initial_state):
    # Create a new annealing problem
    problem = SchedulingProblem(initial_state, toolbox)
    # since our state is just a list, slice is the fastest way to copy
    problem.copy_strategy = "slice"
    best_state, best_energy = problem.anneal()

    return best_state, best_energy


def genetic_algorithm(toolbox, n_medical_staff):
    # Parameters
    population_size = 50
    num_generations = 100
    crossover_prob = 0.8
    mutation_prob = 0.2

    # Register functions with the toolbox
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register(
        "mutate", tools.mutUniformInt, low=0, up=n_medical_staff - 1, indpb=0.1
    )
    toolbox.register("select", tools.selNSGA2)

    # Initialize population
    pop = toolbox.population(n=population_size)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Run the genetic algorithm
    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=num_generations,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    return pop, logbook, hof


from functools import partial


def create_toolbox(
    n_medical_staff,
    resource_requirements,
    staff_preferences,
    solver,
    x,
    operating_rooms,
    case_types,
):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, n_medical_staff - 1)

    # Register a function to create an individual with the proper dimensions
    def init_individual():
        return [
            [
                [random.choice(range(n_medical_staff)) for _ in range(n_medical_staff)]
                for _ in range(len(case_types))
            ]
            for _ in range(len(operating_rooms))
        ]

    toolbox.register(
        "individual", tools.initIterate, creator.Individual, init_individual
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        evaluation_function,
        resource_requirements=resource_requirements,
        staff_preferences=staff_preferences,
    )

    # Use functools.partial to pass the additional arguments to feasible_individual
    feasible_individual_partial = partial(
        feasible_individual,
        model_tuple=(solver, x, operating_rooms, case_types, n_medical_staff),
    )
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible_individual_partial, 9999))

    return toolbox


def main():
    filepath = "../data/TMM Sample Data Site 2 AI.csv"
    n_medical_staff = 10
    df = read_excel_data(filepath)
    (
        operating_rooms,
        case_types,
        resource_requirements,
        staff_preferences,
    ) = process_data(df, n_medical_staff)
    solver, x = create_constraint_programming_model(
        operating_rooms,
        case_types,
        resource_requirements,
        n_medical_staff,
        staff_preferences,
    )

    toolbox = create_toolbox(
        n_medical_staff,
        resource_requirements,
        staff_preferences,
        solver,
        x,
        operating_rooms,
        case_types,
    )

    # Run genetic algorithm
    start_time = time.time()
    pop, logbook, hof = genetic_algorithm(toolbox, n_medical_staff)
    end_time = time.time()
    ga_time = end_time - start_time
    ga_best_solution = hof[0]
    ga_best_energy = toolbox.evaluate(ga_best_solution)

    # Run simulated annealing
    start_time = time.time()
    # Create an initial state for the annealing problem
    initial_state = toolbox.population(n=1)[0]
    # Use simulated annealing to find a good scheduling
    best_state, best_energy = simulated_annealing(toolbox, initial_state)
    end_time = time.time()
    sa_time = end_time - start_time
    sa_best_solution = best_state
    sa_best_energy = best_energy

    print("Genetic Algorithm:")
    print("Time:", ga_time)
    print("Best solution:", ga_best_solution)
    print("Energy of the best solution:", ga_best_energy)

    print("Simulated Annealing:")
    print("Time:", sa_time)
    print("Best solution:", sa_best_solution)
    print("Energy of the best solution:", sa_best_energy)


if __name__ == "__main__":
    main()
