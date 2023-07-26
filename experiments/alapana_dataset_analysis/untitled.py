from math import log2, pow

A4 = 440
C0 = A4*pow(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
def pitch(freq):
    h = round(12*log2(freq/C0))
    octave = h // 12
    n = h % 12
    return name[n] + str(octave)


def myround(x, base=5):
    return base * round(x/base)


def convert(path, timestep, out_path=None, time_col='time', freq_col='freq'):
    df = pd.read_csv(path)

    df[f'time_{timestep}'] = df[time_col].apply(lambda y: myround(y,  timestep))

    df = df[[f'time_{timestep}', freq_col]]

    df['freq_mean'] = df.group_by(f'time_{timestep}').agg('mean').reset_index()

    df['note'] = df['freq_mean'].apply(lambda y: pitch(y))

    df = df[[f'time_{timestep}', 'note']]

    if out_path:
        df.to_csv(out_path, index=False)

    return df


path = ''
out_path = ''
time_col = 'time'
freq_col = 'freq'
timestep = 0.125

df = convert(path, timestep, out_path, time_col, freq_col)