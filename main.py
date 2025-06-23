import hydra
from omegaconf import DictConfig
from src.pipeline.pipeline_builder import PipelineBuilder

@hydra.main(version_base=None,config_path="configs", config_name=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the PathPilot pipeline.
    
    Args:
        cfg: Hydra configuration object
    Returns:
        -
    """
    # Build pipeline from configuration
    pipeline = PipelineBuilder.build(cfg)
    
    # Run pipeline
    pipeline.run()

if __name__ == "__main__":
    main()
