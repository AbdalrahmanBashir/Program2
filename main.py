import random
import numpy as np
from collections import defaultdict


class Facilitator:
    """
    A facilitator is someone who can facilitate an activity.
    """

    def __init__(self, name):
        """
        Initialize a facilitator with a given name.
        """
        self.name = name


class Activity:
    """
    An activity is something that can be facilitated.

    expected_enrollment: The number of students expected to enroll in the
        activity.
    preferred_facilitators: The list of facilitators that the activity
        prefers to be facilitated by.
    other_facilitators: The list of facilitators that the activity is
        willing to be facilitated by, but less than preferred_facilitators.
    """

    def __init__(self, name, expected_enrollment, preferred_facilitators,
                 other_facilitators):
        self.name = name
        self.expected_enrollment = expected_enrollment
        self.preferred_facilitators = preferred_facilitators
        self.other_facilitators = other_facilitators


class Room:
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity


class Assignment:
    """
    An assignment is the combination of an activity, facilitator, room, and
    time slot.
    """

    def __init__(self, activity, facilitator, room, time_slot):
        """
        Initialize an assignment with the given activity, facilitator,
        room, and time slot.

        Args:
            activity: The activity to be assigned.
            facilitator: The facilitator to facilitate the activity.
            room: The room in which the activity is to be held.
            time_slot: The time slot during which the activity is to be held.
        """
        self.activity = activity
        self.facilitator = facilitator
        self.room = room
        self.time_slot = time_slot


def generate_random_schedule(activities, rooms, time_slots, facilitators_):
    """
    Generate a random schedule using the given activities, rooms, time slots,
    and facilitators.

    Args:
        activities: The list of activities to be scheduled.
        rooms: The list of rooms available for scheduling.
        time_slots: The list of time slots available for scheduling.
        facilitators_: The list of facilitators available for scheduling.
    """
    assignments = []

    # Shuffle the list of activities
    # random.shuffle(activities)

    for activity in activities:
        # Choose a random facilitator from the list of available facilitators
        facilitator = random.choice(facilitators_)
        # Choose a random room from the list of available rooms
        available_rooms = [
            room for room in rooms if room.capacity >= activity.expected_enrollment]
        room = random.choice(available_rooms)
        # Choose a random time slot from the list of available time slots
        # time_slots = random.shuffle(time_slots)
        time_slot = random.choice(time_slots)
        # Create a new Assignment object and add it to the list of assignments
        assignments.append(Assignment(
            activity, facilitator.name, room, time_slot))

    return assignments


def calculate_room_penalty(room_capacity, expected_enrollment):
    """
    Calculates a penalty for scheduling an activity in a room with less space
    than expected. The penalty is calculated based on the following:

    - If the room has less than the expected enrollment, the penalty is -0.5
    - If the room has between 3 and 6 times the expected enrollment, the
      penalty is -0.2
    - If the room has more than 6 times the expected enrollment, the penalty
      is -0.4
    - If the room has exactly the expected enrollment, the gain is 0.3
    """
    if room_capacity < expected_enrollment:
        return -0.5
    elif room_capacity > 6 * expected_enrollment:
        return -0.4
    elif room_capacity > 3 * expected_enrollment:
        return -0.2
    else:
        return 0.3


def calculate_facilitator_penalty(facilitator, activity):
    """
    Calculates a penalty for scheduling an activity with a facilitator
    who is not preferred or other.

    The penalty is calculated based on the following:

    - If the facilitator is in the list of preferred facilitators, the
      gain is 0.5
    - If the facilitator is Tyler and there are 2 or less preferred
      facilitators, the gain is 0
    - If the facilitator is in the list of other facilitators, the gain
      is 0.2
    - If the facilitator is not in either list, the penalty is -0.1
    """
    preferred_facilitators = activity.preferred_facilitators
    other_facilitators = activity.other_facilitators

    if facilitator in preferred_facilitators:
        return 0.5
    elif facilitator == "Tyler" and len(preferred_facilitators) <= 2:
        return 0
    elif facilitator in other_facilitators:
        return 0.2
    else:
        return -0.1


def calculate_facilitator_load_penalty(facilitator_activities, schedule):
    """
    Calculates a penalty for scheduling too many activities for a single
    facilitator and for consecutive time slots. The penalty is calculated based on the following:

    - If a facilitator has exactly one activity, the gain is 0.2
    - If a facilitator has more than one activity, the penalty is -0.2

    However, if the facilitator is Tyler, the penalty is adjusted based on
    the number of activities scheduled for Tyler.

    - If Tyler has 4 or more activities, the penalty is -0.5
    - If Tyler has 2 activities, the penalty is -0.4
    - Otherwise, the penalty is -0.2

    Additionally, if any facilitator has consecutive time slots, the penalty is adjusted based on activity-specific rules.
    """
    total_penalty = 0

    for facilitator, count in facilitator_activities.items():
        if count == 1:
            total_penalty += 0.2
        elif count > 1:
            if facilitator == "Tyler":
                if count > 4:
                    total_penalty -= 0.5
                elif 1 < count <= 2:
                    total_penalty -= 0.4
                else:
                    total_penalty -= 0.2
            else:
                total_penalty -= 0.2

    # Check for consecutive time slots
    for activity_index in range(len(schedule) - 1):
        current_activity = schedule[activity_index]
        next_activity = schedule[activity_index + 1]

        if current_activity.facilitator == next_activity.facilitator:
            time_difference = abs(
                current_activity.time_slot - next_activity.time_slot)
            # Check if consecutive time slots
            if time_difference == 0:
                total_penalty -= 0.25
            elif time_difference >= 1:
                total_penalty += 0.25

    return total_penalty


def handle_sla100_adjustments(sla100_activities):
    """
    Calculates a penalty for SLA 100 activities that have time conflicts.

    If two SLA 100 activities have the same time slot, the penalty is -0.5.
    If two SLA 100 activities are more than 4 time slots apart, the gain is +0.5.
    """
    penalty = 0

    for i, assignment in enumerate(sla100_activities):
        for other_activity in sla100_activities[i + 1:]:
            time_difference = abs(assignment.time_slot -
                                  other_activity.time_slot)
            if time_difference == 0:
                # print(f"{assignment.activity.name} and {other_activity.activity.name} have a time conflict.")
                penalty -= 0.5
            elif time_difference >= 4:
                # print(f"{assignment.activity.name} and {other_activity.activity.name} are more than 4 time slots apart.")
                penalty += 0.5

    return penalty


def handle_sla191_adjustments(sla191_activities):
    """
    Calculates a penalty for SLA 191 activities that have time conflicts.

    If two SLA 191 activities have the same time slot, the penalty is -0.5.
    If two SLA 191 activities are more than 4 time slots apart, the gain is +0.5.
    """
    penalty = 0

    for i, assignment in enumerate(sla191_activities):
        for other_activity in sla191_activities[i + 1:]:
            if assignment.time_slot == other_activity.time_slot:
                # print(f"{assignment.activity.name} and {other_activity.activity.name} have a time conflict.")
                penalty -= 0.5
            elif abs(assignment.time_slot - other_activity.time_slot) >= 4:
                # print(f"{assignment.activity.name} and {other_activity.activity.name} are more than 4 time slots apart.")
                penalty += 0.5

    return penalty


def handle_sla_adjustments(schedule):
    """
    Calculates a penalty for SLA 100 and SLA 191 activities based on their time conflicts and room assignments.
    The penalties are as follows:

    * If two SLA 100 activities have the same time slot, the penalty is -0.5.
    * If two SLA 100 activities are more than 4 time slots apart, the gain is +0.5.
    * If two SLA 191 activities have the same time slot, the penalty is -0.5.
    * If two SLA 191 activities are more than 4 time slots apart, the gain is +0.5.
    * If there is a time slot between SLA 101 and SLA 191 that is separated by 1 hour
      and one activity is in Roman or Beach and the other isn't, the penalty is -0.4.
    * If there is a time slot between SLA 101 and SLA 191 that is separated by 1 hour, the gain is +0.5.
    * If there is a time slot between SLA 101 and SLA 191 that is the same time slot, the penalty is -0.25.
    """
    penalty = 0

    sla100_activities = [
        assignment for assignment in schedule if assignment.activity.name.startswith("SLA100")]
    sla191_activities = [
        assignment for assignment in schedule if assignment.activity.name.startswith("SLA191")]

    # Adjustments for SLA 100 activities
    penalty += handle_sla100_adjustments(sla100_activities)

    # Adjustments for SLA 191 activities
    penalty += handle_sla191_adjustments(sla191_activities)

    # Check for consecutive time slots between SLA 101 and SLA 191
    for sla101_activity in sla100_activities:
        for sla191_activity in sla191_activities:
            time_difference = abs(
                sla101_activity.time_slot - sla191_activity.time_slot)
            if time_difference == 1:
                # Check if one activity is in Roman or Beach, and the other isn’t
                if ("Roman" in sla101_activity.room.name or "Beach" in sla101_activity.room.name) != \
                   ("Roman" in sla191_activity.room.name or "Beach" in sla191_activity.room.name):
                    # print("is subtracting 0.4")
                    penalty -= 0.4
                else:
                    # print("is adding 0.5")
                    penalty += 0.5
            elif time_difference == 2:
                # Separated by 1 hour
                penalty += 0.25
            elif time_difference == 0:
                # Same time slot
                penalty -= 0.25

    return penalty


def calculate_activity_fitness(assignment):
    """
    Calculates the fitness score for a particular activity assignment.

    The penalty is calculated based on room capacity and expected enrollment, as well as
    facilitator availability and preferences.
    """
    activity = assignment.activity
    facilitator = assignment.facilitator
    room_capacity = assignment.room.capacity

    penalty = calculate_room_penalty(room_capacity, activity.expected_enrollment) + \
        calculate_facilitator_penalty(facilitator, activity)
    return penalty


def fitness(schedule):
    """
    Calculates the fitness score for a particular schedule.

    The fitness score is calculated based on the following:

    1. The penalty for each activity assignment based on room capacity, expected
       enrollment, facilitator availability, and preferences.
    2. The penalty for SLA 101 and SLA 191 being in consecutive time slots, if applicable
    3. The number of activities each facilitator is assigned

    The fitness score is normalized so that the best possible score is 1 and the worst is 0.
    """

    unnormalized_fitness_score = 0  # Initialize the unnormalized fitness score to 0
    # Create a dictionary to keep track of the number of activities each facilitator is assigned
    facilitator_activities = defaultdict(int)

    for assignment in schedule:  # Iterate over each activity assignment
        # Calculate the penalty for this activity assignment
        activity_penalty = calculate_activity_fitness(assignment)
        # Add the activity penalty to the unnormalized fitness score
        unnormalized_fitness_score += activity_penalty
        # Increment the number of activities the facilitator is assigned
        facilitator_activities[assignment.facilitator] += 1

    # Apply any adjustments to the fitness score based on SLA conflicts
    unnormalized_fitness_score += handle_sla_adjustments(schedule)
    unnormalized_fitness_score += calculate_facilitator_load_penalty(
        facilitator_activities, schedule)  # Apply a penalty for each facilitator being assigned too many activities

    # Return the unnormalized fitness score
    return unnormalized_fitness_score


def softmax(fitness_scores):
    """
    The softmax function maps a vector of real-valued numbers to a vector of
    real numbers in the range [0, 1] that sum up to 1.

    The formula for the softmax function is:

        softmax(x_1, x_2, ..., x_n) = exp(x_1) / (exp(x_1) + exp(x_2) + ... + exp(x_n))

    where x_1, x_2, ..., x_n are the input numbers, and n is the number of input numbers.

    The softmax function is calculated as follows:

    1. First, we subtract the maximum value from each element of the
       fitness scores vector, to prevent overflow. This is done because
       the exponential function grows very quickly, and the maximum value
       in the vector could be very large.

    2. Next, we compute the exponential of each element in the adjusted
       vector. This gives us a new vector of positive numbers.

    3. We then add up all the elements in this vector to get a single
       scalar value, which we will call the denominator.

    4. Finally, we divide each element of the input vector by the
       denominator, which gives us the final vector of softmax values.
    """

    # Find the maximum value in the fitness_scores
    max_score = np.max(fitness_scores)

    # Subtract the maximum value from each element of fitness_scores
    adjusted_scores = fitness_scores - max_score

    # Compute the softmax function using the adjusted values
    exp_scores = np.exp(adjusted_scores)
    prob_distribution = exp_scores / np.sum(exp_scores)
    return prob_distribution


def selection(population, fitness_scores, k):
    """
    Selects the top k schedules from the population based on their
    fitness scores. The selection is done using a probabilistic approach
    based on the softmax function. The probability of a schedule being
    selected is proportional to its fitness score, which is a positive
    real number that represents how well the schedule solves the
    scheduling problem. The softmax function maps the fitness scores
    to a vector of real numbers in the range [0, 1] that sum up to 1,
    which is necessary for the np.random.choice function to work
    correctly.

    The algorithm first computes the softmax of the fitness scores
    using the softmax function defined above. It then generates k
    random indices from the range of the length of the population
    using np.random.choice with the probability distribution being the
    softmax of the fitness scores. Finally, it returns the schedules
    corresponding to the selected indices.
    """

    prob_distribution = softmax(fitness_scores)
    selected_indices = np.random.choice(
        len(population), size=k, p=prob_distribution, replace=True)
    return [population[i] for i in selected_indices]


def crossover(schedule1, schedule2, crossover_rate=1.0):
    if random.random() > crossover_rate:
        # No crossover, return parents as offspring
        return schedule1[:], schedule2[:]

    # Calculate the number of crossover points based on the length of schedules
    num_genes = min(len(schedule1), len(schedule2))
    # Ensure at least one crossover point
    num_crossover_points = min(2, num_genes - 1)

    # Generate random crossover points
    crossover_points = sorted(random.sample(
        range(num_genes), num_crossover_points))

    offspring1, offspring2 = [], []
    last_point = 0
    for point in crossover_points:
        if (point - last_point) % 2 == 0:
            offspring1.extend(schedule1[last_point:point])
            offspring2.extend(schedule2[last_point:point])
        else:
            offspring1.extend(schedule2[last_point:point])
            offspring2.extend(schedule1[last_point:point])
        last_point = point

    # Add remaining genes
    offspring1.extend(schedule1[last_point:])
    offspring2.extend(schedule2[last_point:])

    return offspring1, offspring2


def mutation(schedule, rooms, time_slots, facilitators, mutation_rate, generation, num_generations):
    mutated_schedule = schedule.copy()

    # Calculate scaled mutation rate based on generation number
    scaled_mutation_rate = mutation_rate * (1 - generation / num_generations)

    # Check if the assignment is already optimal
    if random.random() >= scaled_mutation_rate:
        return mutated_schedule

    for assignment in mutated_schedule:
        mutation_type = random.choice(['facilitator', 'room', 'time_slot'])
        if mutation_type == 'facilitator':
            available_facilitators = [
                f for f in facilitators if f.name != assignment.facilitator
            ]
            if available_facilitators:
                assignment.facilitator = random.choice(
                    facilitators).name
        elif mutation_type == 'room':
            available_rooms = [
                room for room in rooms if room.name != assignment.room
            ]
            # and room.capacity >= assignment.activity.expected_enrollment
            if available_rooms:
                assignment.room = random.choice(rooms)
        elif mutation_type == 'time_slot':
            available_time_slots = [
                t for t in time_slots if t != assignment.time_slot
            ]
            if available_time_slots:
                assignment.time_slot = random.choice(time_slots)

    return mutated_schedule


def find_best_schedule(population):
    return max(population, key=fitness)


def genetic_algorithm_optimized(population_size, activities, rooms, time_slots, facilitators, initial_mutation_rate, num_generations):
    """
    Runs a genetic algorithm to find the optimal schedule for the given
    parameters. The algorithm starts with a random population of schedules,
    calculates the fitness of each schedule in the current population,
    selects parents for crossover and mutation, creates new offspring by
    applying crossover and mutation to the selected parents, replaces the
    current population with the offspring, and repeats the process for the
    specified number of generations. It returns the best schedule found
    and its unnormalized fitness score.
    initial_mutation_rate: The initial mutation rate for the algorithm
    num_generations: The number of generations the algorithm should run
    """

    # Generate initial population
    population = [generate_random_schedule(
        activities, rooms, time_slots, facilitators) for _ in range(population_size)]

    # Find the best schedule in the initial population
    best_schedule = find_best_schedule(population)
    best_unnormalized_fitness = fitness(best_schedule)

    mutation_rate = initial_mutation_rate
    min_mutation_rate = 0.0003125
    max_mutation_rate = initial_mutation_rate

    print(
        f"initial Generation: New best fitness score: {best_unnormalized_fitness}")

    for generation in range(num_generations):
        # Calculate fitness of each schedule in current population
        fitness_scores = [fitness(schedule) for schedule in population]

        # Select parents for crossover and mutation
        selected = selection(population, fitness_scores, population_size)

        # Create offspring from selected parents
        offspring = []
        for i in range(0, population_size, 2):
            offspring.extend(crossover(selected[i], selected[i+1]))

        # Mutate offspring
        mutated_offspring = [mutation(
            schedule, rooms, time_slots, facilitators, mutation_rate, generation, num_generations) for schedule in offspring]

        # Replace current population with offspring
        population = mutated_offspring

        # Find the best schedule in the current population
        current_best_schedule = find_best_schedule(population)
        current_unnormalized_fitness = fitness(
            current_best_schedule)

        # Update the best schedule if the new one has a higher unnormalized fitness
        if current_unnormalized_fitness > best_unnormalized_fitness:
            print(
                f"Generation {generation+1}: New best fitness score: {current_unnormalized_fitness}")

            best_schedule = current_best_schedule
            best_unnormalized_fitness = current_unnormalized_fitness

        # Adjust mutation rate for next generation
        mutation_rate *= 0.75
        mutation_rate = max(min_mutation_rate, min(
            max_mutation_rate, mutation_rate))

    return [best_schedule, best_unnormalized_fitness]


def write_schedules_to_file(best_schedule, best_fitness, filename="best_schedule.txt"):
    # Sort the best schedule by time slot
    best_schedule_sorted = sorted(best_schedule, key=lambda x: x.time_slot)
    with open(filename, "w") as file:
        # Single schedule case
        file.write("Best Schedule is:\n")
        file.write("Fitness Score: {:.5f}\n\n".format(
            best_fitness))
        file.write("{:<12} {:<20} {:<15} {:<20} {:<10}\n".format(
            "Activity", "Facilitator", "Room", "Time Slot", "Enrollment"))
        file.write("-" * 75 + "\n")
        for assignment in best_schedule_sorted:

            file.write("{:<12} {:<20} {:<15} {:<20} {:<10}\n".format(
                assignment.activity.name,
                assignment.facilitator,
                assignment.room.name,
                assignment.time_slot,
                assignment.activity.expected_enrollment
            ))
        file.write("\n" + "-" * 75 + "\n\n")


def main():
    population_size = 500
    num_generations = 100
    mutation_rate = 0.01

    activities = [
        Activity("SLA100A", 50, ["Glen", "Lock", "Banks", "Zeldin"], [
                 "Numen", "Richards"]),
        Activity("SLA100B", 50, ["Glen", "Lock", "Banks", "Zeldin"], [
                 "Numen", "Richards"]),
        Activity("SLA191A", 50, ["Glen", "Lock", "Banks", "Zeldin"], [
                 "Numen", "Richards"]),
        Activity("SLA191B", 50, ["Glen", "Lock", "Banks", "Zeldin"], [
                 "Numen", "Richards"]),
        Activity("SLA201", 50, ["Glen", "Banks", "Zeldin", "Shaw"], [
                 "Numen", "Richards", "Singer"]),
        Activity("SLA291", 50, ["Lock", "Banks", "Zeldin", "Singer"], [
                 "Numen", "Richards", "Shaw", "Tyler"]),
        Activity("SLA303", 60, ["Glen", "Zeldin", "Banks"], [
                 "Numen", "Singer", "Shaw"]),
        Activity("SLA304", 25, ["Glen", "Banks", "Tyler"], [
                 "Numen", "Singer", "Shaw", "Richards", "Uther", "Zeldin"]),
        Activity("SLA394", 20, ["Tyler", "Singer"], ["Richards", "Zeldin"]),
        Activity("SLA449", 60, ["Tyler", "Singer",
                 "Shaw"], ["Zeldin", "Uther"]),
        Activity("SLA451", 100, ["Tyler", "Singer", "Shaw"], [
                 "Zeldin", "Uther", "Richards", "Banks"])
    ]

    rooms = [
        Room("Slater 003", 50),  # 45
        Room("Roman 216", 30),  # 30
        Room("Loft 206", 75),
        Room("Roman 201", 50),
        Room("Loft 310", 108),
        Room("Beach 201", 60),
        Room("Beach 301", 75)
    ]

    facilitators = [
        Facilitator("Lock"),
        Facilitator("Glen"),
        Facilitator("Banks"),
        Facilitator("Richards"),
        Facilitator("Shaw"),
        Facilitator("Singer"),
        Facilitator("Uther"),
        Facilitator("Tyler"),
        Facilitator("Numen"),
        Facilitator("Zeldin")
    ]

    time_slots = [10, 11, 12, 13, 14, 15]

    best_schedule, best_fitness = genetic_algorithm_optimized(
        population_size, activities, rooms, time_slots, facilitators, mutation_rate,  num_generations)
    write_schedules_to_file(best_schedule, best_fitness)


if __name__ == "__main__":
    main()
