from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import click
from examples.instrument import run_example

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.group()
def cli():
    pass

@cli.command(
    name='run_example',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_cmd(example_module_name, example_argv):
    return run_example(example_module_name, example_argv)

cli.add_command(run_example_cmd)

def main():
    return cli()

if __name__ == "__main__":
    main()
