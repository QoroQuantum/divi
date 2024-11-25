def counts_to_expectation_value(results):
    eigenvalue = 0
    total_shots = 0

    for key, val in results.items():
        if key.count("1") % 2 == 1:
            eigenvalue += -val
        else:
            eigenvalue += val
        total_shots += val

    return eigenvalue / total_shots
