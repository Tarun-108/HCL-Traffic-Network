import traci

def get_lane_queue_length(lane_id):
    return traci.lane.getLastStepHaltingNumber(lane_id)

def get_lane_waiting_time(lane_id):
    return traci.lane.getWaitingTime(lane_id)

def get_lane_occupancy(lane_id):
    return traci.lane.getLastStepOccupancy(lane_id)

def get_lane_vehicle_count(lane_id):
    return traci.lane.getLastStepVehicleNumber(lane_id)

def get_lane_mean_speed(lane_id):
    """Returns the mean speed of vehicles on a lane (in m/s)."""
    return traci.lane.getLastStepMeanSpeed(lane_id)

def get_lane_vehicle_ids(lane_id):
    """Returns list of vehicle IDs on the lane."""
    return traci.lane.getLastStepVehicleIDs(lane_id)

def get_traffic_light_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def set_traffic_light_phase(tls_id, phase):
    traci.trafficlight.setPhase(tls_id, phase)

def get_traffic_light_state(tls_id):
    return traci.trafficlight.getRedYellowGreenState(tls_id)

def get_incoming_lanes(tls_id):
    return traci.trafficlight.getControlledLanes(tls_id)

def get_total_queue_length(tls_id):
    lanes = get_incoming_lanes(tls_id)
    return sum(get_lane_queue_length(lane) for lane in lanes)

def get_total_waiting_time(tls_id):
    lanes = get_incoming_lanes(tls_id)
    return sum(get_lane_waiting_time(lane) for lane in lanes)

def get_total_vehicle_count(tls_id):
    lanes = get_incoming_lanes(tls_id)
    return sum(get_lane_vehicle_count(lane) for lane in lanes)

def get_total_lane_occupancy(tls_id):
    lanes = get_incoming_lanes(tls_id)
    return sum(get_lane_occupancy(lane) for lane in lanes) / len(lanes)

def get_observation_for_tls(tls_id):
    """Returns a state vector for RL agent based on queue length, waiting time, and occupancy."""
    lanes = get_incoming_lanes(tls_id)
    return [get_lane_queue_length(l) for l in lanes] + \
           [get_lane_waiting_time(l) for l in lanes] + \
           [get_lane_occupancy(l) for l in lanes]

def get_reward(tls_id, weight_queue=1.0, weight_wait=1.0):
    """
    Computes the reward for a given traffic light ID.
    Negative of weighted sum of queue length and waiting time (since we want to minimize these).
    """
    total_queue = get_total_queue_length(tls_id)
    total_wait = get_total_waiting_time(tls_id)
    
    # You can scale or normalize as needed
    reward = - (weight_queue * total_queue + weight_wait * total_wait)
    return reward

def get_reward_normalized(tls_id):
    lanes = get_incoming_lanes(tls_id)
    total_lanes = len(lanes)

    total_queue = get_total_queue_length(tls_id)
    total_wait = get_total_waiting_time(tls_id)

    avg_queue = total_queue / total_lanes
    avg_wait = total_wait / total_lanes

    reward = - (avg_queue + avg_wait)
    return reward
