"""
File: data.py
Author: Chuncheng Zhang
Date: 2025-05-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    The data class of the operation.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-05-14 ------------------------
# Requirements and constants
import mne
from enum import Enum
from .logging import logger


# %% ---- 2025-05-14 ------------------------
# Function and class
class Events(Enum):
    Hand = '1',  # '手'
    Wrist = '2',  # '腕'
    Elbow = '3',  # '肘'
    Shoulder = '4',  # '肩'
    Rest = '5',  # '静息'


class MEGPart:
    # ch_names in the data
    data_ch_names = ['BG1-4504', 'BG2-4504', 'BG3-4504', 'BP1-4504', 'BP2-4504', 'BP3-4504', 'BR1-4504', 'BR2-4504', 'BR3-4504', 'G11-4504', 'G12-4504', 'G13-4504', 'G22-4504', 'G23-4504', 'P11-4504', 'P12-4504', 'P13-4504', 'P22-4504', 'P23-4504', 'Q11-4504', 'Q12-4504', 'Q13-4504', 'Q22-4504', 'Q23-4504', 'R11-4504', 'R12-4504', 'R13-4504', 'R22-4504', 'MLC11-4504', 'MLC12-4504', 'MLC13-4504', 'MLC14-4504', 'MLC15-4504', 'MLC16-4504', 'MLC17-4504', 'MLC21-4504', 'MLC22-4504', 'MLC23-4504', 'MLC24-4504', 'MLC25-4504', 'MLC31-4504', 'MLC32-4504', 'MLC41-4504', 'MLC42-4504', 'MLC51-4504', 'MLC52-4504', 'MLC53-4504', 'MLC54-4504', 'MLC55-4504', 'MLC61-4504', 'MLC62-4504', 'MLC63-4504', 'MLF11-4504', 'MLF12-4504', 'MLF13-4504', 'MLF14-4504', 'MLF21-4504', 'MLF22-4504', 'MLF23-4504', 'MLF24-4504', 'MLF25-4504', 'MLF31-4504', 'MLF32-4504', 'MLF33-4504', 'MLF34-4504', 'MLF35-4504', 'MLF41-4504', 'MLF42-4504', 'MLF43-4504', 'MLF44-4504', 'MLF45-4504', 'MLF46-4504', 'MLF51-4504', 'MLF52-4504', 'MLF53-4504', 'MLF54-4504', 'MLF56-4504', 'MLF61-4504', 'MLF62-4504', 'MLF63-4504', 'MLF64-4504', 'MLF65-4504', 'MLF66-4504', 'MLF67-4504', 'MLO11-4504', 'MLO12-4504', 'MLO13-4504', 'MLO14-4504', 'MLO21-4504', 'MLO22-4504', 'MLO23-4504', 'MLO24-4504', 'MLO31-4504', 'MLO32-4504', 'MLO33-4504', 'MLO34-4504', 'MLO41-4504', 'MLO42-4504', 'MLO43-4504', 'MLO44-4504', 'MLO51-4504', 'MLO52-4504', 'MLO53-4504', 'MLP11-4504', 'MLP12-4504', 'MLP21-4504', 'MLP22-4504', 'MLP23-4504', 'MLP31-4504', 'MLP32-4504', 'MLP33-4504', 'MLP34-4504', 'MLP35-4504', 'MLP41-4504', 'MLP42-4504', 'MLP43-4504', 'MLP44-4504', 'MLP45-4504', 'MLP51-4504', 'MLP52-4504', 'MLP53-4504', 'MLP54-4504', 'MLP55-4504', 'MLP56-4504', 'MLP57-4504', 'MLT11-4504', 'MLT12-4504', 'MLT13-4504', 'MLT14-4504', 'MLT15-4504', 'MLT16-4504', 'MLT21-4504', 'MLT22-4504', 'MLT23-4504', 'MLT24-4504', 'MLT25-4504', 'MLT26-4504', 'MLT27-4504', 'MLT31-4504', 'MLT32-4504', 'MLT33-4504', 'MLT34-4504', 'MLT35-4504', 'MLT36-4504', 'MLT37-4504', 'MLT41-4504', 'MLT42-4504', 'MLT43-4504', 'MLT44-4504', 'MLT45-4504', 'MLT46-4504', 'MLT47-4504',
                     'MLT51-4504', 'MLT52-4504', 'MLT53-4504', 'MLT54-4504', 'MLT55-4504', 'MLT56-4504', 'MLT57-4504', 'MRC11-4504', 'MRC12-4504', 'MRC13-4504', 'MRC14-4504', 'MRC15-4504', 'MRC16-4504', 'MRC17-4504', 'MRC21-4504', 'MRC22-4504', 'MRC23-4504', 'MRC24-4504', 'MRC25-4504', 'MRC31-4504', 'MRC32-4504', 'MRC41-4504', 'MRC42-4504', 'MRC51-4504', 'MRC52-4504', 'MRC53-4504', 'MRC54-4504', 'MRC55-4504', 'MRC61-4504', 'MRC62-4504', 'MRC63-4504', 'MRF11-4504', 'MRF12-4504', 'MRF13-4504', 'MRF14-4504', 'MRF21-4504', 'MRF22-4504', 'MRF23-4504', 'MRF24-4504', 'MRF25-4504', 'MRF31-4504', 'MRF32-4504', 'MRF33-4504', 'MRF34-4504', 'MRF35-4504', 'MRF41-4504', 'MRF42-4504', 'MRF43-4504', 'MRF44-4504', 'MRF45-4504', 'MRF46-4504', 'MRF51-4504', 'MRF52-4504', 'MRF53-4504', 'MRF54-4504', 'MRF55-4504', 'MRF56-4504', 'MRF61-4504', 'MRF62-4504', 'MRF63-4504', 'MRF64-4504', 'MRF65-4504', 'MRF66-4504', 'MRF67-4504', 'MRO11-4504', 'MRO12-4504', 'MRO13-4504', 'MRO14-4504', 'MRO21-4504', 'MRO22-4504', 'MRO23-4504', 'MRO24-4504', 'MRO31-4504', 'MRO32-4504', 'MRO33-4504', 'MRO34-4504', 'MRO41-4504', 'MRO42-4504', 'MRO43-4504', 'MRO44-4504', 'MRO51-4504', 'MRO52-4504', 'MRO53-4504', 'MRP11-4504', 'MRP12-4504', 'MRP21-4504', 'MRP22-4504', 'MRP23-4504', 'MRP31-4504', 'MRP32-4504', 'MRP33-4504', 'MRP34-4504', 'MRP35-4504', 'MRP41-4504', 'MRP42-4504', 'MRP43-4504', 'MRP44-4504', 'MRP45-4504', 'MRP51-4504', 'MRP52-4504', 'MRP53-4504', 'MRP54-4504', 'MRP55-4504', 'MRP56-4504', 'MRP57-4504', 'MRT11-4504', 'MRT12-4504', 'MRT13-4504', 'MRT14-4504', 'MRT15-4504', 'MRT16-4504', 'MRT21-4504', 'MRT22-4504', 'MRT24-4504', 'MRT25-4504', 'MRT26-4504', 'MRT27-4504', 'MRT31-4504', 'MRT32-4504', 'MRT33-4504', 'MRT34-4504', 'MRT35-4504', 'MRT36-4504', 'MRT37-4504', 'MRT41-4504', 'MRT42-4504', 'MRT43-4504', 'MRT44-4504', 'MRT45-4504', 'MRT46-4504', 'MRT47-4504', 'MRT51-4504', 'MRT52-4504', 'MRT53-4504', 'MRT54-4504', 'MRT55-4504', 'MRT56-4504', 'MRT57-4504', 'MZC01-4504', 'MZC02-4504', 'MZC03-4504', 'MZC04-4504', 'MZF01-4504', 'MZF02-4504', 'MZF03-4504', 'MZO01-4504', 'MZO02-4504', 'MZO03-4504', 'MZP01-4504']

# ! Scaling dict(eeg=1e6, mag=1e15, grad=1e13)


def mk_ch_dig_mapping():
    _table = {
        # L1
        'fz': 308,
        'f1': 329,
        'f3': 307,
        'f5': 328,
        'f2': 330,
        'f4': 309,
        'f6': 331,
        # L2
        'fcz': 338,
        'fc1': 337,
        'fc3': 336,
        'fc5': 335,
        'fc2': 339,
        'fc4': 340,
        'fc6': 341,
        # L3
        'cz': 313,
        'c1': 346,
        'c3': 312,
        'c5': 345,
        'c2': 347,
        'c4': 314,
        'c6': 348,
        # L4
        'cpz': 353,
        'cp1': 352,
        'cp3': 351,
        'cp5': 350,
        'cp2': 354,
        'cp4': 355,
        'cp6': 356,
        # L5
        'pz': 318,
        'p1': 359,
        'p3': 317,
        'p5': 358,
        'p2': 360,
        'p4': 319,
        'p6': 361,
    }

    def _ch_name(s):
        s = s.strip()
        return s[:-1].upper() + s[-1]

    return {_ch_name(k): v for k, v in _table.items()}


def mk_good_eeg_ch_names():
    ch_names = '''
        f5,  f3,  f1,  fz,  f2,  f4,  f6,
        fc5, fc3, fc1, fcz, fc2, fc4, fc6,
        c5,  c3,  c1,  cz,  c2,  c4,  c6,
        cp5, cp3, cp1, cpz, cp2, cp4, cp6,
        p5,  p3,  p1,  pz,  p2,  p4,  p6
    '''

    def _ch_name(s):
        s = s.strip()
        return s[:-1].upper() + s[-1]

    return [_ch_name(e) for e in ch_names.split(',')]


class EEGPart:
    # Only the first 35 out of 64 channels are used
    # 35 = 5 x 7
    ch_names = mk_good_eeg_ch_names()
    standard_montage_name = 'standard_1020'

    # ch_names in the data
    data_ch_names = ['EEG001-4504', 'EEG002-4504', 'EEG003-4504', 'EEG004-4504', 'EEG005-4504', 'EEG006-4504', 'EEG007-4504', 'EEG008-4504', 'EEG009-4504', 'EEG010-4504', 'EEG011-4504', 'EEG012-4504', 'EEG013-4504', 'EEG014-4504', 'EEG015-4504', 'EEG016-4504', 'EEG017-4504', 'EEG018-4504', 'EEG019-4504', 'EEG020-4504', 'EEG021-4504', 'EEG022-4504', 'EEG023-4504', 'EEG024-4504', 'EEG025-4504', 'EEG026-4504', 'EEG027-4504', 'EEG028-4504', 'EEG029-4504', 'EEG030-4504', 'EEG031-4504', 'EEG032-4504',
                     'EEG033-4504', 'EEG034-4504', 'EEG035-4504', 'EEG036-4504', 'EEG037-4504', 'EEG038-4504', 'EEG039-4504', 'EEG040-4504', 'EEG041-4504', 'EEG042-4504', 'EEG043-4504', 'EEG044-4504', 'EEG045-4504', 'EEG046-4504', 'EEG047-4504', 'EEG048-4504', 'EEG049-4504', 'EEG050-4504', 'EEG051-4504', 'EEG052-4504', 'EEG053-4504', 'EEG054-4504', 'EEG055-4504', 'EEG056-4504', 'EEG057-4504', 'EEG058-4504', 'EEG059-4504', 'EEG060-4504', 'EEG061-4504', 'EEG062-4504', 'EEG063-4504', 'EEG064-4504']


class MyData:
    raw: mne.io.Raw
    noise_raw: mne.io.Raw
    events: list
    event_id: dict
    meg_epochs: mne.Epochs
    eeg_epochs: mne.Epochs
    empty_room_raw: mne.io.Raw
    empty_room_projs: list

    def setattr(self, **kwargs):
        '''
        Set the known attribute.

        :param **kwargs: The attr: value to be set.
        '''
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
                logger.warning(f'Override {k} = {v}')
            else:
                setattr(self, k, v)
                logger.info(f'Set {k} = {v}')

    def add_proj(self):
        '''
        Add the empty room projection to the raw data.
        ! Since the empty room data only contains the MEG channels, the EEG channels are not included.
        '''
        empty_room_raw = self.noise_raw
        empty_room_projs = mne.compute_proj_raw(empty_room_raw)
        self.raw.add_proj(empty_room_projs)
        self.empty_room_raw = empty_room_raw
        self.empty_room_projs = empty_room_projs
        logger.info(f'Add empty room proj: {empty_room_projs}')

    def generate_epochs(self, **kwargs):
        '''
        Generate eeg- and meg-epochs from raw with given kwargs

        :param **kwargs: The kwargs in the mne.Epochs(raw, **kwargs).
        '''
        # MEG Epochs
        # ! Dynamic acquire meg ch_names, since they may change.
        epochs = mne.Epochs(self.raw, self.events,
                            self.event_id, picks=['meg'])
        info = epochs.info

        meg_ch_names = epochs.ch_names
        epochs = mne.Epochs(
            self.raw, self.events, self.event_id, picks=meg_ch_names, **kwargs)
        ch_names_with_4504 = epochs.ch_names
        rename = {name: name.split('-')[0] for name in ch_names_with_4504}
        epochs.rename_channels(rename)
        logger.info(f'Rename MEG channels: {rename}')
        self.meg_epochs = epochs
        logger.info(f'Loaded MEG Epochs {self.meg_epochs}')

        # EEG Epochs
        ch_names = EEGPart.data_ch_names[:len(EEGPart.ch_names)]
        epochs = mne.Epochs(
            self.raw, self.events, self.event_id, picks=ch_names, **kwargs)

        rename = {src: dst for src, dst in zip(ch_names, EEGPart.ch_names)}
        epochs.rename_channels(rename)
        logger.info(f'Rename EEG channels: {rename}')

        # Set EEG chs location
        use_dig = True
        if use_dig:
            ch_dig_mapping = mk_ch_dig_mapping()
            # Use digits as EEG chs location
            for ch in epochs.info['chs']:
                trg_ident = ch_dig_mapping[ch['ch_name']]
                for dig in info['dig']:
                    if dig['ident'] == trg_ident:
                        ch['loc'][:3] = dig['r']
                        logger.debug(f'Locate {ch}\'loc into {dig}')
                        break
        else:
            # Use standard montage
            epochs.set_montage(
                EEGPart.standard_montage_name, on_missing='warn')
            logger.info(f'Set EEG montage: {EEGPart.standard_montage_name}')

        self.eeg_epochs = epochs
        logger.info(f'Loaded EEG Epochs {self.eeg_epochs}')

# %% ---- 2025-05-14 ------------------------
# Play ground


# %% ---- 2025-05-14 ------------------------
# Pending


# %% ---- 2025-05-14 ------------------------
# Pending
