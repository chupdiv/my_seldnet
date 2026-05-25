from dataclasses import dataclass, InitVar

@dataclass
class Annotation:
    """Единая разметка одного звукового события.

    Attributes:
        onset:  Начало события, секунды (относительно начала отрезка).
        offset: Конец события, секунды (относительно начала отрезка).
        label:  Имя класса (например ``"drone"``, ``"Voices"``, ``"Dog bark"``).
        duration: (InitVar) Общая длительность отрезка.
    """
    onset: float
    offset: float
    label: str
    duration: InitVar[float | None] = None  # Поле, которое не сохраняется в экземпляре, но нужно для валидации

    def __post_init__(self, duration: float | None):
        if self.onset >= self.offset:
            raise ValueError(
                f"Время начала (onset={self.onset}) должно быть меньше времени конца (offset={self.offset})."
            )
        if self.onset < 0:
            raise ValueError(f"Onset ({self.onset}) не может быть отрицательным.")

        #Проверка принадлежности промежутку [0, duration], если задана длительность
        if duration is not None:
            if self.offset > duration:
                raise ValueError(
                    f"Событие ({self.onset}-{self.offset}) выходит за границы длительности ({duration})."
                )

    def contains(self, time: float) -> bool:
        """Проверяет, находится ли временная метка внутри события (левая граница включена, правая не включена).
        """
        return self.onset <= time < self.offset

    def __contains__(self, time: float) -> bool:
        return self.contains(time)            
    


