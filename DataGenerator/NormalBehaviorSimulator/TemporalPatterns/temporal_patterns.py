import math

class TemporalPatterns:
    def __init__(self):
        pass

    def circadian_factor(self, timestamp):
        # Business hours (9am-5pm): High activity
        # Night (10pm-6am): Low activity
        day_seconds = timestamp % 86400
        hour = day_seconds / 3600
        if 9 <= hour <= 17:  # 9am-5pm
            factor = 1.0 + 0.5 * math.sin((hour - 13) * math.pi / 4)  # Peak around 1pm
        elif 22 <= hour or hour <= 6:  # 10pm-6am
            factor = 0.3  # Low activity
        else:
            factor = 0.7  # Transition periods
        return factor

    def weekly_factor(self, timestamp):
        # Monday: Ramp-up (80% baseline)
        # Tuesday-Thursday: Peak (100% baseline)
        # Friday: Wind-down (90% baseline)
        # Weekend: Minimal (20-30% baseline)
        day_of_week = (timestamp // 86400) % 7  # 0=Monday, 6=Sunday
        if day_of_week == 0:  # Monday
            factor = 0.8
        elif 1 <= day_of_week <= 3:  # Tue-Thu
            factor = 1.0
        elif day_of_week == 4:  # Friday
            factor = 0.9
        else:  # Weekend
            factor = 0.25  # Average of 20-30%
        return factor

    def seasonal_factor(self, timestamp):
        # Monthly cycles (billing, payroll)
        # Quarterly trends (business cycles)
        # Annual patterns (holidays, fiscal year)
        # For simplicity, simulate monthly and annual
        month_seconds = 30 * 24 * 3600  # ~30 days
        year_seconds = 365 * 24 * 3600  # ~365 days

        monthly_phase = (timestamp % month_seconds) / month_seconds * 2 * math.pi
        annual_phase = (timestamp % year_seconds) / year_seconds * 2 * math.pi

        monthly_factor = 1 + 0.1 * math.sin(monthly_phase)  # 10% monthly variation
        annual_factor = 1 + 0.05 * math.sin(annual_phase)   # 5% annual variation

        return monthly_factor * annual_factor

    def apply_temporal(self, base_value, timestamp):
        circadian = self.circadian_factor(timestamp)
        weekly = self.weekly_factor(timestamp)
        seasonal = self.seasonal_factor(timestamp)
        return base_value * circadian * weekly * seasonal