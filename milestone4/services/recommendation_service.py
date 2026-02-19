def generate_recommendations(df):

    avg_device = df.groupby("appliance")["energy"].mean()
    overall_avg = df["energy"].mean()

    recs = []

    for device, value in avg_device.items():
        if value > overall_avg:
            recs.append(f"{device} consumes above average energy. Optimize usage time.")
        else:
            recs.append(f"{device} operates efficiently compared to others.")

    return recs
