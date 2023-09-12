from typing import Optional

from funlib.geometry import Roi
import numpy as np
import dacapo
import click
import logging
from dacapo.experiments.datasplits.datasets.dataset import Dataset

from dacapo.experiments.tasks.post_processors.post_processor_parameters import (
    PostProcessorParameters,
)


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command()
@click.option(
    "-r", "--run-name", required=True, type=str, help="The NAME of the run to train."
)
def train(run_name):
    dacapo.train(run_name)


@cli.command()
@click.option(
    "-r", "--run-name", required=True, type=str, help="The name of the run to validate."
)
@click.option(
    "-i",
    "--iteration",
    required=True,
    type=int,
    help="The iteration at which to validate the run.",
)
def validate(run_name, iteration):
    dacapo.validate(run_name, iteration)


@cli.command()
@click.option(
    "-r", "--run", required=True, type=str, help="The name of the run to use."
)
@click.option(
    "-ic",
    "--input_container",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("-id", "--input_dataset", required=True, type=str)
@click.option(
    "-oc", "--output_container", required=True, type=click.Path(file_okay=False)
)
@click.option("-vd", "--validation_dataset", type=str, default=None)
@click.option("-c", "--criterion", default="voi")
@click.option("-i", "--iteration", type=int, default=None)
@click.option("-p", "--parameters", type=str, default=None)
@click.option(
    "-roi",
    "--roi",
    type=str,
    required=False,
    help="The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]",
)
@click.option("-w", "--num_cpu_workers", type=int, default=30)
@click.option("-dt", "--output_dtype", type=str, default="uint8")
def apply(
    run_name: str,
    input_container: Path or str,
    input_dataset: str,
    output_container: Path or str,
    validation_dataset: Optional[Dataset or str] = None,
    criterion: Optional[str] = "voi",
    iteration: Optional[int] = None,
    parameters: Optional[PostProcessorParameters or str] = None,
    roi: Optional[Roi or str] = None,
    num_cpu_workers: int = 4,
    output_dtype: Optional[np.dtype or str] = np.uint8,
):
    dacapo.apply(
        run_name,
        input_container,
        input_dataset,
        output_container,
        validation_dataset,
        criterion,
        iteration,
        parameters,
        roi,
        num_cpu_workers,
        output_dtype,
    )
