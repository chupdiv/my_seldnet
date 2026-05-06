import os
import numpy as np

from .config import Config


class ProtoModel():
    def __init__(self, path:str, device:str='cpu',**kwargs) -> None:
        self.name = os.path.split(path)[1]
        name_parts = self.name.lower().split('.') 
        self.active = True
        if len(name_parts) > 1:
            #Метод неактивен, если каталог имеет расширение '.disable' или '.off'
            self.active = (name_parts[-1] not in ['off','disable'])
            # Расширение имени каталога не является частью имени модели
            self.name = '.'.join(name_parts[:-1])    

        self.model_type = None # Тип модели 'localizer', 'detector', и. т.д.
        self.path = path
        self.device = device
        self.load_config(path)
        self.mic_pos = None
        self.mic_pairs = None
        self.set_prob_threshold(0.)

    def load_config(self, path) -> None:
        self.path = path
        configfile = self._get_configfile(path)
        if not os.path.isfile(configfile):
            raise FileNotFoundError(f"Файл конфигурации модели не найден: {configfile}")
        cfg = Config(configfile)
        self._update(cfg)

    def _update(self, cfg) -> None:
            self.tags = cfg.get_section('tags')
            self.caption = self.tags.get('caption',self.name)
            self.description = self.tags.get('description','')
            self.model_class = self.tags.get('class','')
            self.sort_order = int(self.tags.get('order',10000))

            model_type = self.tags.get('type','')
            self.model_type = self._parse_modeltype(model_type)

            files_section = cfg.get_section('files')
            self.files = {key: self._safe_join_model_path(fname) for key, fname in files_section.items()}
            self.settings = cfg.get_config(exclude={'tags','files'})

    def _safe_join_model_path(self, relative_path: str) -> str:
        """
        Защита от выхода за пределы директории модели через '../' в model.conf.
        """
        base = os.path.abspath(self.path)
        candidate = os.path.abspath(os.path.join(base, relative_path))
        if candidate != base and not candidate.startswith(base + os.sep):
            raise ValueError(f"Некорректный путь в конфиге модели: '{relative_path}' (выходит за пределы '{base}')")
        return candidate
    
    def _parse_modeltype(self,st:str):
        L = st.split(',')
        return {s.strip() for s in L}
    
    def _get_configfile(self,path:str) -> str:
        configfile = 'model.conf'
        return os.path.join(path, configfile)
    
    def load(self):
        pass
         
    def predict(self, S):
        pass

    def set_prob_threshold(self, threshold:float):
        self.prob_threshold = np.float32(threshold)

    def set_mics_positions(self, mic_pos):
        if mic_pos is not None:
            self.mic_pos = np.asarray(mic_pos, dtype=np.float32)
    
    def set_mics_pairs(self, pairs):
        if pairs is not None:
            self.mic_pairs = np.asarray(pairs, dtype=int)
         
             


        

