from abc import ABC,abstractmethod

class yoga_exercise(ABC):

    @abstractmethod
    def pose_positions(self,frames,draw):
        pass

    @abstractmethod
    def pose_landmarks(self,frames,draw):
        pass