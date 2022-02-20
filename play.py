# play.py 
import argparse
from trainer import Trainer

#pasrser = argparse.ArgumentParser()


trainer = Trainer(field_width=120, field_height=120)
trainer.preview(render_fps=60)

