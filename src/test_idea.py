from policy.base_schedules.constant import ConstantSchedule
from privacy.gdp_privacy import GDPPrivacyParameters

# Hard-coded settings
EPS = 3.0
DELTA = 0.000001
P = 0.00416666666667
T = 750

SIGMA_INIT = 2.12873
CLIP_INIT = 3.9843063


def main():
    privacy_params = GDPPrivacyParameters(eps=EPS, delta=DELTA, p=P, T=T)

    noise_schedule = ConstantSchedule(value=SIGMA_INIT, T=T)
    clip_schedule = ConstantSchedule(value=CLIP_INIT, T=T)

    sigmas = noise_schedule.get_valid_schedule()
    clips = clip_schedule.get_valid_schedule()

    print("Before projection:")
    print(f"  sigmas: {noise_schedule.value}")
    print(f"  clips:  {clip_schedule.value}")
    print(f"  expenditure: {privacy_params.compute_expenditure(sigmas, clips):.4f}")
    print(f"  budget (mu/p)^2: {(privacy_params.mu / privacy_params.p) ** 2:.4f}")

    proj_sigmas, proj_clips = privacy_params.project_sigma_and_clip(sigmas, clips)
    noise_schedule = ConstantSchedule.from_projection(noise_schedule, proj_sigmas)
    clip_schedule = ConstantSchedule.from_projection(clip_schedule, proj_clips)

    print("\nAfter projection:")
    print(f"  sigmas: {noise_schedule.value}")
    print(f"  clips:  {clip_schedule.value}")
    print(f"  expenditure: {privacy_params.compute_expenditure(proj_sigmas, proj_clips):.4f}")


if __name__ == "__main__":
    main()
