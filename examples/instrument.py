"""Functions for instrumenting and running softlearning examples.

This package contains functions, which allow seamless runs of examples in
different modes (e.g. locally, in google compute engine, or ec2).


There are two types of functions in this file:
1. run_example_* methods, which run the experiments by invoking
    `tune.run_experiments` function.
2. launch_example_* methods, which are helpers function to submit an
    example to be run in the cloud. In practice, these launch a cluster,
    and then run the `run_example_cluster` method with the provided
    arguments and options.
"""

import importlib
import multiprocessing
import os
import uuid
from pprint import pformat
import pdb
from softlearning.misc.utils import datetimestamp, PROJECT_PATH

def _normalize_trial_resources(resources, cpu, gpu, extra_cpu, extra_gpu):
    if resources is None:
        resources = {}

    if cpu is not None:
        resources['cpu'] = cpu

    if gpu is not None:
        resources['gpu'] = gpu

    if extra_cpu is not None:
        resources['extra_cpu'] = extra_cpu

    if extra_gpu is not None:
        resources['extra_gpu'] = extra_gpu

    return resources


def add_command_line_args_to_variant_spec(variant_spec, command_line_args):
    variant_spec['run_params'].update({
        'checkpoint_frequency': (
            command_line_args.checkpoint_frequency
            if command_line_args.checkpoint_frequency is not None
            else variant_spec['run_params'].get('checkpoint_frequency', 0)
        ),
        'checkpoint_at_end': (
            command_line_args.checkpoint_at_end
            if command_line_args.checkpoint_at_end is not None
            else variant_spec['run_params'].get('checkpoint_at_end', True)
        ),
    })

    variant_spec['restore'] = command_line_args.restore

    return variant_spec


def generate_experiment(trainable_class, variant_spec, command_line_args):
    params = variant_spec.get('algorithm_params')
    resources_per_trial = _normalize_trial_resources(
        command_line_args.resources_per_trial,
        command_line_args.trial_cpus,
        command_line_args.trial_gpus,
        command_line_args.trial_extra_cpus,
        command_line_args.trial_extra_gpus)

    experiment_id = params.get('exp_name')

    #### add pool_load_max_size to experiment_id
    if 'pool_load_max_size' in variant_spec['algorithm_params']['kwargs']:
        max_size = variant_spec['algorithm_params']['kwargs']['pool_load_max_size']
        experiment_id = '{}_{}e3'.format(experiment_id, int(max_size/1000))
    ####

    variant_spec = add_command_line_args_to_variant_spec(
        variant_spec, command_line_args)

    if command_line_args.video_save_frequency is not None:
        assert 'algorithm_params' in variant_spec
        variant_spec['algorithm_params']['kwargs']['video_save_frequency'] = (
            command_line_args.video_save_frequency)

    experiment = {
        'run': trainable_class,
        'resources_per_trial': resources_per_trial,
        'config': variant_spec,
        'num_samples': command_line_args.num_samples,
        'upload_dir': command_line_args.upload_dir,
        'checkpoint_freq': (
            variant_spec['run_params']['checkpoint_frequency']),
        'checkpoint_at_end': (
            variant_spec['run_params']['checkpoint_at_end']),
        'restore': command_line_args.restore,  # Defaults to None
    }

    return experiment_id, experiment


def unique_cluster_name(args):
    cluster_name_parts = (
        datetimestamp(''),
        str(uuid.uuid4())[:6],
        args.domain,
        args.task
    )
    cluster_name = "-".join(cluster_name_parts).lower()
    return cluster_name


def confirm_yes_no(prompt):
    # raw_input returns the empty string for "enter"
    yes = {'yes', 'ye', 'y'}
    no = {'no', 'n'}

    choice = input(prompt).lower()
    while True:
        if choice in yes:
            return True
        elif choice in no:
            exit(0)
        else:
            print("Please respond with 'yes' or 'no'.\n(yes/no)")
        choice = input().lower()


def run_example(example_module_name, example_argv, local_mode=False):
    """Run example locally, potentially parallelizing across cpus/gpus."""
    example_module = importlib.import_module(example_module_name)

    example_args = example_module.get_parser().parse_args(example_argv)
    variant_spec = example_module.get_variant_spec(example_args)

    variant_spec['environment_params']['evaluation'] = variant_spec['environment_params']['training']

    trainable_class = example_module.get_experiment(example_args)

    experiment_id, experiment = generate_experiment(
        trainable_class, variant_spec, example_args)
    experiments = {experiment_id: experiment}

    trainable_class._setup(trainable_class, experiment['config'])
    trainable_class._train(trainable_class)
