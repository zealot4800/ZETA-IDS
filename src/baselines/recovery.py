"""Read durable journal prefixes without modifying the original run."""
import json


def journal(path):
    if not path.exists():
        return
    with path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            # Only newline-terminated records are committed. A torn final write
            # can be retried; corruption in a committed record must be reported.
            if not line.endswith(b'\n'):
                if stream.read(1):
                    raise ValueError(f'Invalid journal line {number}: {path}')
                return
            try:
                yield json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f'Invalid journal line {number}: {path}') from exc


def copy_journal(source, target):
    if source.exists():
        with target.open('x') as output:
            for record in journal(source):
                output.write(json.dumps(record, allow_nan=False) + '\n')
