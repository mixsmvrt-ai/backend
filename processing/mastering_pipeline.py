
import subprocess


def run(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def ai_master(input_mix: str, output_master: str, target_lufs: str = "-14") -> str:
    """Apply gentle master EQ, compression and limiting towards a LUFS target.

    This approximates the MASTER BUS section of MIX_AND_MASTER_FLOW_DSP:
    low-shelf and air lift, low-ratio bus compression, and a limiter with
    ceiling at -1 dBTP feeding EBU R128 loudness normalisation.
    """

    cmd = (
        f'ffmpeg -y -i "{input_mix}" -af '
        '"equalizer=f=100:t=q:w=1.1:g=1,'  # low-region lift ~80–120 Hz
        'equalizer=f=13000:t=q:w=1.3:g=1,'  # air lift ~12–16 kHz
        'acompressor=threshold=-18dB:ratio=1.7:attack=50:release=180,'
        'alimiter=limit=0.89,'
        f'loudnorm=I={target_lufs}:TP=-1.0:LRA=11" '
        f'"{output_master}"'
    )

    run(cmd)
    return output_master
