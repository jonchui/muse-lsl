import numpy as np
import matplotlib
from scipy.signal import lfilter, lfilter_zi, firwin, welch
from pylsl import StreamInlet, resolve_byprop
import seaborn as sns
from .constants import VIEW_BUFFER, VIEW_SUBSAMPLE, LSL_SCAN_TIMEOUT, LSL_EEG_CHUNK


CHANNEL_DESCRIPTIONS = {
    'TP9': 'left ear/temple',
    'AF7': 'left forehead',
    'AF8': 'right forehead',
    'TP10': 'right ear/temple',
    'Right AUX': 'right aux',
}

CHANNEL_COLORS = {
    'TP9': 'tab:blue',
    'AF7': 'tab:orange',
    'AF8': 'tab:green',
    'TP10': 'tab:red',
    'Right AUX': 'tab:purple',
}

DEFAULT_CHANNEL_COLORS = (
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
)
CHANNEL_ROW_PADDING = 1.0

POWER_WINDOWS = (1, 5, 10)

# The viewer's display filter is 1-40 Hz, so these powers are approximate
# sub-bands of the signal shown on screen.
EEG_BANDS = (
    ('delta', 1, 4),
    ('theta', 4, 8),
    ('alpha', 8, 12),
    ('beta', 12, 30),
    ('gamma', 30, 40),
)


def integrate_band_power(y, x, axis=0):
    if hasattr(np, 'trapezoid'):
        integrate = np.trapezoid
    else:
        integrate = np.trapz
    return integrate(y, x, axis=axis)


def view(window, scale, refresh, figure, backend, version=1):
    matplotlib.use(backend)
    sns.set(style="whitegrid")

    figsize = np.int16(figure.split('x'))

    print("Looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        raise(RuntimeError("Can't find EEG stream."))
    print("Start acquiring data.")

    fig, axes = matplotlib.pyplot.subplots(1, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(left=0.24, right=0.70)
    lslv = LSLViewer(streams[0], fig, axes, window, scale, refresh)
    fig.canvas.mpl_connect('close_event', lslv.stop)

    help_str = """
                toggle filter : d
                reconnect/reset stream : r
                toogle full screen : f
                zoom out : /
                zoom in : *
                increase time scale : -
                decrease time scale : +
               """
    print(help_str)
    lslv.start()
    matplotlib.pyplot.show()


class LSLViewer():
    def __init__(self, stream, fig, axes, window, scale, refresh, dejitter=True):
        """Init"""
        self.stream = stream
        self.window = window
        self.scale = scale
        self.refresh = refresh
        self.dejitter = dejitter
        self.inlet = StreamInlet(stream, max_chunklen=LSL_EEG_CHUNK)
        self.filt = True
        self.subsample = VIEW_SUBSAMPLE

        info = self.inlet.info()
        description = info.desc()

        self.sfreq = info.nominal_srate()
        self.n_samples = int(self.sfreq * self.window)
        self.n_chan = info.channel_count()

        ch = description.child('channels').first_child()
        ch_names = [ch.child_value('label')]

        for i in range(self.n_chan - 1):
            ch = ch.next_sibling()
            ch_names.append(ch.child_value('label'))

        self.ch_names = ch_names
        self.ch_labels = [self.describe_channel(name) for name in ch_names]
        self.ch_colors = [
            self.channel_color(ii, name) for ii, name in enumerate(ch_names)
        ]

        fig.canvas.mpl_connect('key_press_event', self.OnKeypress)
        fig.canvas.mpl_connect('button_press_event', self.onclick)

        self.fig = fig
        self.axes = axes

        sns.despine(left=True)

        self.data = np.zeros((self.n_samples, self.n_chan))
        self.times = np.arange(-self.window, 0, 1. / self.sfreq)
        signal_std = np.std(self.data, axis=0)
        lines = []

        for ii in range(self.n_chan):
            line, = axes.plot(self.times[::self.subsample],
                              self.data[::self.subsample, ii] - ii,
                              color=self.ch_colors[ii],
                              lw=1)
            lines.append(line)
        self.lines = lines

        axes.set_ylim(self.y_limits())
        ticks = np.arange(0, -self.n_chan, -1)

        axes.set_xlabel('Time (s)')
        axes.set_title('Muse EEG: traces filtered 1-40 Hz; band power shown at right')
        axes.xaxis.grid(False)
        axes.set_yticks(ticks)

        self.set_channel_tick_labels(signal_std)

        self.metrics_text = axes.text(
            1.03, 0.98, '',
            transform=axes.transAxes,
            va='top',
            ha='left',
            family='monospace',
            fontsize=9,
            bbox={
                'boxstyle': 'round,pad=0.5',
                'facecolor': 'white',
                'edgecolor': '0.75',
                'alpha': 0.9,
            })

        self.display_every = max(1, int(self.refresh / (12 / self.sfreq)))

        self.af = [1.0]
        self.last_metrics_error = None
        self._reset_buffers()

    def describe_channel(self, ch_name):
        description = CHANNEL_DESCRIPTIONS.get(ch_name)
        if description:
            return '%s (%s)' % (ch_name, description)
        return ch_name

    def channel_color(self, index, ch_name):
        default_color = DEFAULT_CHANNEL_COLORS[
            index % len(DEFAULT_CHANNEL_COLORS)]
        return CHANNEL_COLORS.get(ch_name, default_color)

    def y_limits(self):
        bottom_channel = -(self.n_chan - 1)
        return bottom_channel - CHANNEL_ROW_PADDING, CHANNEL_ROW_PADDING

    def set_channel_tick_labels(self, signal_std):
        ticks_labels = ['%s - SD %.2fuV' % (self.ch_labels[ii],
                                            signal_std[ii])
                        for ii in range(self.n_chan)]
        self.axes.set_yticklabels(ticks_labels)
        for tick, color in zip(self.axes.get_yticklabels(), self.ch_colors):
            tick.set_color(color)

    def _format_metric(self, value):
        if np.isnan(value):
            return '   -- '
        if value >= 1000:
            return '%6.0f' % value
        return '%6.1f' % value

    def _reset_buffers(self):
        self.n_samples = int(self.sfreq * self.window)
        self.times = np.arange(-self.window, 0, 1. / self.sfreq)
        self.data = np.zeros((self.n_samples, self.n_chan))
        self.data_f = np.zeros((self.n_samples, self.n_chan))
        self.metric_samples = int(self.sfreq * max(POWER_WINDOWS))
        self.metric_data_f = np.zeros((self.metric_samples, self.n_chan))
        self.metric_valid_samples = 0
        self.bf = firwin(32, np.array([1, 40]) / (self.sfreq / 2.), width=0.05,
                         pass_zero=False)
        zi = lfilter_zi(self.bf, self.af)
        self.filt_state = np.tile(zi, (self.n_chan, 1)).transpose()

    def reconnect(self):
        print("Looking for an EEG stream to reconnect...")
        streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)
        if len(streams) == 0:
            print("Can't find EEG stream. Keep the stream process running.")
            return

        inlet = StreamInlet(streams[0], max_chunklen=LSL_EEG_CHUNK)
        info = inlet.info()
        n_chan = info.channel_count()
        if n_chan != self.n_chan:
            print("EEG channel count changed; close and reopen the viewer.")
            return

        self.stream = streams[0]
        self.inlet = inlet
        self.sfreq = info.nominal_srate()
        self._reset_buffers()
        print("Reconnected EEG viewer.")

    def _band_powers(self, data):
        """Return average band power across channels for one window."""
        if data.shape[0] < max(4, int(self.sfreq)):
            return {band[0]: np.nan for band in EEG_BANDS}

        data = data - data.mean(axis=0, keepdims=True)
        nperseg = min(data.shape[0], int(self.sfreq * 2))
        freqs, psd = welch(data, fs=self.sfreq, axis=0, nperseg=nperseg)
        powers = {}

        for name, fmin, fmax in EEG_BANDS:
            band = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(band):
                powers[name] = np.nan
                continue
            channel_power = integrate_band_power(
                psd[band], freqs[band], axis=0)
            powers[name] = np.nanmean(channel_power)

        return powers

    def _metrics_summary(self, signal_std):
        band_rows = {name: [] for name, _, _ in EEG_BANDS}

        for seconds in POWER_WINDOWS:
            sample_count = int(self.sfreq * seconds)
            if self.metric_valid_samples < sample_count:
                powers = {band[0]: np.nan for band in EEG_BANDS}
            else:
                window_data = self.metric_data_f[-sample_count:]
                powers = self._band_powers(window_data)

            for name, _, _ in EEG_BANDS:
                band_rows[name].append(powers[name])

        lines = [
            'Band power uV^2',
            'avg across sensors',
            'band      1s     5s    10s',
        ]

        for name, _, _ in EEG_BANDS:
            values = ''.join(self._format_metric(value)
                             for value in band_rows[name])
            lines.append('%-5s %s' % (name, values))

        lines.extend(['', 'Sensor SD uV'])
        for name, value in zip(self.ch_names, signal_std):
            lines.append('%-9s %5.1f' % (name, value))

        return '\n'.join(lines)

    def update_plot(self):
        updated = False
        for _ in range(self.display_every):
            samples, timestamps = self.inlet.pull_chunk(timeout=0.0,
                                                        max_samples=LSL_EEG_CHUNK)
            if not timestamps:
                break

            if self.dejitter:
                timestamps = np.float64(np.arange(len(timestamps)))
                timestamps /= self.sfreq
                timestamps += self.times[-1] + 1. / self.sfreq
            self.times = np.concatenate([self.times, timestamps])
            self.n_samples = int(self.sfreq * self.window)
            self.times = self.times[-self.n_samples:]
            self.data = np.vstack([self.data, samples])
            self.data = self.data[-self.n_samples:]
            filt_samples, self.filt_state = lfilter(
                self.bf, self.af,
                samples,
                axis=0, zi=self.filt_state)
            self.data_f = np.vstack([self.data_f, filt_samples])
            self.data_f = self.data_f[-self.n_samples:]
            self.metric_data_f = np.vstack([self.metric_data_f, filt_samples])
            self.metric_data_f = self.metric_data_f[-self.metric_samples:]
            self.metric_valid_samples = min(
                self.metric_samples,
                self.metric_valid_samples + len(samples))
            updated = True

        if updated:
            if self.filt:
                plot_data = self.data_f
            elif not self.filt:
                plot_data = self.data - self.data.mean(axis=0)
            for ii in range(self.n_chan):
                self.lines[ii].set_xdata(self.times[::self.subsample] -
                                         self.times[-1])
                self.lines[ii].set_ydata(plot_data[::self.subsample, ii] /
                                         self.scale - ii)
                signal_std = np.std(plot_data, axis=0)

            self.set_channel_tick_labels(signal_std)
            try:
                self.metrics_text.set_text(self._metrics_summary(signal_std))
                self.last_metrics_error = None
            except Exception as err:
                message = '%s: %s' % (err.__class__.__name__, err)
                if message != self.last_metrics_error:
                    print('Band-power metrics unavailable: %s' % message)
                    self.last_metrics_error = message
                self.metrics_text.set_text(
                    'Band power unavailable\n%s\nraw traces still running' %
                    message)
            self.axes.set_xlim(-self.window, 0)
            self.fig.canvas.draw_idle()

        return self.started

    def onclick(self, event):
        print((event.button, event.x, event.y, event.xdata, event.ydata))

    def OnKeypress(self, event):
        if event.key == '/':
            self.scale *= 1.2
        elif event.key == '*':
            self.scale /= 1.2
        elif event.key == '+':
            self.window += 1
        elif event.key == '-':
            if self.window > 1:
                self.window -= 1
        elif event.key == 'd':
            self.filt = not(self.filt)
        elif event.key == 'r':
            self.reconnect()

    def start(self):
        self.started = True
        self.timer = self.fig.canvas.new_timer(
            interval=max(1, int(self.refresh * 1000)))
        self.timer.add_callback(self.update_plot)
        self.timer.start()

    def stop(self, close_event=None):
        self.started = False
        if hasattr(self, 'timer'):
            self.timer.stop()
