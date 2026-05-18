#!/usr/bin/env python3
import argparse
import csv
import json
import os
import select
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pylsl import StreamInfo, StreamOutlet, local_clock


def iso_local(timestamp):
    return datetime.fromtimestamp(timestamp).astimezone().isoformat()


def iso_utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def write_manifest(path, manifest):
    with path.open('w') as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write('\n')


def push_marker(outlet, writer, file_handle, start_time, label):
    now = time.time()
    lsl_timestamp = local_clock()
    outlet.push_sample([label], lsl_timestamp)
    writer.writerow({
        'wall_time_iso': iso_local(now),
        'unix_time': '%.6f' % now,
        'session_seconds': '%.3f' % (now - start_time),
        'lsl_timestamp': '%.6f' % lsl_timestamp,
        'label': label,
    })
    file_handle.flush()
    print('[%7.1fs] marker: %s' % (now - start_time, label))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Record EEG and typed event markers into a timestamped session folder.')
    parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Recording duration in seconds. Default: 3600.')
    parser.add_argument(
        '--out-dir',
        default='recordings',
        help='Directory where session folders are created. Default: recordings.')
    parser.add_argument(
        '--label',
        default='muse-session',
        help='Human-readable session label saved to the manifest.')
    parser.add_argument(
        '--notes',
        default='',
        help='Optional notes saved to the manifest.')
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    session_id = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d-%H.%M.%S')
    session_dir = Path(args.out_dir) / ('session-%s' % session_id)
    session_dir.mkdir(parents=True, exist_ok=False)

    eeg_csv = session_dir / 'eeg.csv'
    markers_csv = session_dir / 'markers.csv'
    manifest_path = session_dir / 'manifest.json'
    record_log = session_dir / 'record.log'

    manifest = {
        'session_id': session_id,
        'label': args.label,
        'notes': args.notes,
        'duration_seconds': args.duration,
        'local_start_iso': iso_local(start_time),
        'utc_start_iso': iso_utc(start_time),
        'unix_start': start_time,
        'lsl_local_clock_start': local_clock(),
        'eeg_csv': str(eeg_csv),
        'markers_csv': str(markers_csv),
        'record_log': str(record_log),
        'loom_alignment': (
            "Start Loom near session start, then type 'loom_start' here. "
            "Loom elapsed time can be aligned to markers.csv session_seconds."),
    }
    write_manifest(manifest_path, manifest)

    info = StreamInfo('Markers', 'Markers', 1, 0, 'string',
                      'muse-session-markers-%s' % session_id)
    outlet = StreamOutlet(info)

    with markers_csv.open('w', newline='') as marker_file, record_log.open('w') as log_file:
        fieldnames = [
            'wall_time_iso',
            'unix_time',
            'session_seconds',
            'lsl_timestamp',
            'label',
        ]
        writer = csv.DictWriter(marker_file, fieldnames=fieldnames)
        writer.writeheader()

        print('Session folder: %s' % session_dir)
        print('Start Loom now if you want video, then type: loom_start')
        print('Type markers like: open_messages, deep_work, break, distracted')
        print('Type /quit to stop marker entry; EEG recording will finish/stop cleanly.')
        push_marker(outlet, writer, marker_file, start_time, 'session_start')

        record_cmd = [
            sys.executable,
            '-u',
            '-m',
            'muselsl',
            'record',
            '--duration',
            str(args.duration),
            '--filename',
            str(eeg_csv),
        ]
        recorder = subprocess.Popen(
            record_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd())

        try:
            deadline = start_time + args.duration
            while time.time() < deadline:
                if recorder.poll() is not None:
                    print('Recorder exited; see %s' % record_log)
                    break

                remaining = max(0, deadline - time.time())
                readable, _, _ = select.select([sys.stdin], [], [], min(1.0, remaining))
                if not readable:
                    continue

                label = sys.stdin.readline()
                if label == '':
                    break

                label = label.strip()
                if not label:
                    continue
                if label == '/quit':
                    break

                push_marker(outlet, writer, marker_file, start_time, label)
        except KeyboardInterrupt:
            print('\nStopping session...')
        finally:
            push_marker(outlet, writer, marker_file, start_time, 'session_end')
            if recorder.poll() is None:
                recorder.terminate()
                try:
                    recorder.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    recorder.kill()
                    recorder.wait()

            end_time = time.time()
            manifest['local_end_iso'] = iso_local(end_time)
            manifest['utc_end_iso'] = iso_utc(end_time)
            manifest['unix_end'] = end_time
            manifest['actual_duration_seconds'] = end_time - start_time
            manifest['recorder_exit_code'] = recorder.returncode
            write_manifest(manifest_path, manifest)

    print('Done. EEG, markers, and manifest are in: %s' % session_dir)


if __name__ == '__main__':
    main()
