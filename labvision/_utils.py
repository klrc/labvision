
import labvision


def init(project: str):
    labvision.utils.init(project=project)
    labvision.auto.init(project=project)
    labvision.server.init()
