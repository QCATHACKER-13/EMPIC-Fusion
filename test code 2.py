def calculate_reaction_probability(cross_section, relative_velocity, time_step):
    reaction_rate = cross_section * relative_velocity
    probability = reaction_rate * time_step
    return probability

def decide_reaction(probability):
    random_number = random.uniform(0, 1)
    return random_number < probability

# Main simulation loop
for each time step:
    for each particle pair:
        cross_section = calculate_cross_section(particle_properties)
        probability = calculate_reaction_probability(cross_section, relative_velocity, time_step)
        
        if decide_reaction(probability):
            perform_nuclear_reaction(particle_properties)
