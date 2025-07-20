from datetime import datetime

import numpy as np
import pandas as pd


class CometMLWriter:
    """
    Class for experiment tracking via CometML.

    See https://www.comet.com/docs/v2/.
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        workspace=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        try:
            import comet_ml

            comet_ml.login()

            self.run_id = run_id

            resume = False
            if project_config["trainer"].get("resume_from") is not None:
                resume = True

            if resume:
                if mode == "offline":
                    exp_class = comet_ml.ExistingOfflineExperiment
                else:
                    exp_class = comet_ml.ExistingExperiment

                self.exp = exp_class(experiment_key=self.run_id)
            else:
                if mode == "offline":
                    exp_class = comet_ml.OfflineExperiment
                else:
                    exp_class = comet_ml.Experiment

                self.exp = exp_class(
                    project_name=project_name,
                    workspace=workspace,
                    experiment_key=self.run_id,
                    log_code=kwargs.get("log_code", False),
                    log_graph=kwargs.get("log_graph", False),
                    auto_metric_logging=kwargs.get("auto_metric_logging", False),
                    auto_param_logging=kwargs.get("auto_param_logging", False),
                )
                self.exp.set_name(run_name)
                self.exp.log_parameters(parameters=project_config)

            self.comel_ml = comet_ml

        except ImportError:
            logger.warning("For use comet_ml install it via \n\t pip install comet_ml")

        self.step = 0
        # the mode is usually equal to the current partition name
        # used to separate Partition1 and Partition2 metrics
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        return f"{object_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        self.exp.log_model(
            name="checkpoints", file_or_folder=checkpoint_path, overwrite=True
        )

    def add_scalar(self, scalar_name, scalar):
        self.exp.log_metrics(
            {
                self._object_name(scalar_name): scalar,
            },
            step=self.step,
        )

    def add_scalars(self, scalars):
        self.exp.log_metrics(
            {
                self._object_name(scalar_name): scalar
                for scalar_name, scalar in scalars.items()
            },
            step=self.step,
        )

    def add_image(self, image_name, image):
        self.exp.log_image(
            image_data=image, name=self._object_name(image_name), step=self.step
        )

    def add_audio(self, audio_name, audio, sample_rate=None):
        audio = audio.detach().cpu().numpy().T
        self.exp.log_audio(
            file_name=self._object_name(audio_name),
            audio_data=audio,
            sample_rate=sample_rate,
            step=self.step,
        )

    def add_text(self, text_name, text):
        self.exp.log_text(
            text=text, step=self.step, metadata={"name": self._object_name(text_name)}
        )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        values_for_hist = values_for_hist.detach().cpu().numpy()
        self.exp.log_histogram_3d(
            values=values_for_hist, name=self._object_name(hist_name), step=self.step
        )

    def add_table(self, table_name, table: pd.DataFrame):
        self.exp.set_step(self.step)
        self.exp.log_table(
            filename=self._object_name(table_name) + ".csv",
            tabular_data=table,
            headers=True,
        )

    def add_images(self, image_names, images):
        raise NotImplementedError()

    def add_pr_curve(self, curve_name, curve):
        raise NotImplementedError()

    def add_embedding(self, embedding_name, embedding):
        raise NotImplementedError()
