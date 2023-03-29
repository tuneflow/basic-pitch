from __future__ import annotations

from tuneflow_py import TuneflowPlugin, Song, ParamDescriptor, WidgetType, TrackType, InjectSource, Track, Clip, TuneflowPluginTriggerData, ClipAudioDataInjectData, pitch_to_hz
from typing import Any
import tempfile
import traceback
import tensorflow as tf
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from pretty_midi import PrettyMIDI, Instrument, Note
from math import floor, ceil


basic_pitch_model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))


class BasicPitchTranscribe(TuneflowPlugin):
    @staticmethod
    def provider_id():
        return "andantei"

    @staticmethod
    def plugin_id():
        return "basic-pitch"

    @staticmethod
    def params(song: Song) -> dict[str, ParamDescriptor]:
        return {
            "clipAudioData": {
                "displayName": {
                    "zh": '音频',
                    "en": 'Audio',
                },
                "defaultValue": None,
                "widget": {
                    "type": WidgetType.NoWidget.value,
                },
                "hidden": True,
                "injectFrom": {
                    "type": InjectSource.ClipAudioData.value,
                    "options": {
                        "clips": "selectedAudioClips",
                        "convert": {
                            "toFormat": "ogg",
                            "options": {
                                "sampleRate": 44100
                            }
                        }
                    }
                }
            },
            "onsetThreshold": {
                "displayName": {
                    "zh": '音符分割粒度',
                    "en": 'Note Segmentation Granularity',
                },
                "description": {
                    "zh": '值越小，越容易检测到新的音符',
                    "en": 'The smaller the value, the easier the model detects a new note',
                },
                "defaultValue": 0.5,
                "widget": {
                    "type": WidgetType.Slider.value,
                    "config": {
                        "minValue": 0.05,
                        "maxValue": 0.95,
                        "step": 0.05,
                    },
                },
            },
            "frameThresh": {
                "displayName": {
                    "zh": '音符生成粒度',
                    "en": 'Note Creation Granularity',
                },
                "description": {
                    "zh": '值越小，越容易生成音符',
                    "en": 'The smaller the value, the easier the model creates a note',
                },
                "defaultValue": 0.3,
                "widget": {
                    "type": WidgetType.Slider.value,
                    "config": {
                        "minValue": 0.05,
                        "maxValue": 0.95,
                        "step": 0.05,
                    },
                },
            },
            "minNoteLen": {
                "displayName": {
                    "zh": '最小音符长度',
                    "en": 'Minimum Note Length',
                },
                "description": {
                    "zh": '创建一个音符需要的最小长度(ms)',
                    "en": 'The minimum length required to create a note, in milliseconds',
                },
                "defaultValue": 11,
                "widget": {
                    "type": WidgetType.Slider.value,
                    "config": {
                        "minValue": 3,
                        "maxValue": 50,
                        "step": 1,
                    },
                },
            },
            "maxPitch": {
                "displayName": {
                    "zh": '音高上限',
                    "en": 'Maximum Allowed Note Pitch',
                },
                "defaultValue": 102,
                "widget": {
                    "type": WidgetType.Pitch.value,
                    "config": {
                        "minAllowedPitch": 28,
                        "maxAllowedPitch": 102,
                    },
                },
            },
            "minPitch": {
                "displayName": {
                    "zh": '音高下限',
                    "en": 'Minimum Allowed Note Pitch',
                },
                "defaultValue": 28,
                "widget": {
                    "type": WidgetType.Pitch.value,
                    "config": {
                        "minAllowedPitch": 28,
                        "maxAllowedPitch": 102,
                    },
                },
            },
        }

    @staticmethod
    def run(song: Song, params: dict[str, Any]):
        trigger: TuneflowPluginTriggerData = params["trigger"]
        trigger_entity_id = trigger["entities"][0]
        track = song.get_track_by_id(trigger_entity_id["trackId"])
        if track is None:
            raise Exception("Cannot find track")
        clip = track.get_clip_by_id(trigger_entity_id["clipId"])
        if clip is None:
            raise Exception("Cannot find clip")
        target_tempo = song.get_tempo_event_at_tick(clip.get_clip_start_tick()).get_bpm()
        clip_audio_data_list: ClipAudioDataInjectData = params["clipAudioData"]
        minFreq = floor(pitch_to_hz(params["minPitch"]))
        maxFreq = ceil(pitch_to_hz(params["maxPitch"]))

        clip_start_time = song.tick_to_seconds(clip.get_clip_start_tick())

        tmp_file = tempfile.NamedTemporaryFile(delete=True, suffix=clip_audio_data_list[0]["audioData"]["format"])
        tmp_file.write(clip_audio_data_list[0]["audioData"]["data"])

        try:
            model_output, midi_data, note_events = predict(
                tmp_file.name, basic_pitch_model, onset_threshold=params["onsetThreshold"],
                frame_threshold=params["frameThresh"],
                minimum_note_length=params["minNoteLen"],
                minimum_frequency=minFreq, maximum_frequency=maxFreq, midi_tempo=target_tempo)
            midi_data: PrettyMIDI = midi_data
            new_midi_track = song.create_track(type=TrackType.MIDI_TRACK, index=song.get_track_index(
                track_id=track.get_id()),
                assign_default_sampler_plugin=True)
            new_midi_clip = new_midi_track.create_midi_clip(
                clip_start_tick=clip.get_clip_start_tick(),
                clip_end_tick=clip.get_clip_end_tick(),
                insert_clip=True)
            for input_track in midi_data.instruments:
                input_track: Instrument = input_track
                for note in input_track.notes:
                    note: Note = note
                    note_start_time = note.start + clip_start_time
                    note_end_time = note.end + clip_start_time
                    print(note, note_start_time, note_end_time)
                    added_note = new_midi_clip.create_note(pitch=note.pitch.item(), velocity=note.velocity, start_tick=song.seconds_to_tick(
                        note_start_time), end_tick=song.seconds_to_tick(note_end_time), update_clip_range=False, resolve_clip_conflict=False)
        except Exception as e:
            print(traceback.format_exc())
        finally:
            tmp_file.close()
