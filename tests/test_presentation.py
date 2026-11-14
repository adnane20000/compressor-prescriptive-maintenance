"""Affichages bornés et graphiques sur mesures SYNTHÉTIQUES uniquement."""

import json

import pandas as pd
import pytest

from metropt_maintenance.presentation import (anomaly_figure, create_synthetic_demo, file_identity,
    mode_figure, preview_measurements, read_alerts, read_json, read_score_view, sensor_figure, workflow_figure)


def test_synthetic_demo_has_only_official_signals_and_explicit_context(tmp_path):
    from metropt_maintenance.ingestion.schema import REQUIRED_COLUMNS
    path=create_synthetic_demo(tmp_path)
    frame=pd.read_csv(path)
    assert set(frame)==set(REQUIRED_COLUMNS) and len(frame)==721
    assert read_json(path.with_suffix('.context.json'))['data_context']=='synthetic_test'
    preview=preview_measurements(path,limit=50)
    assert len(preview)==50 and preview.timestamp.is_monotonic_increasing
    assert 'failure_label' not in preview and 'mode_interprete' in preview
    assert file_identity(path)[1]==path.stat().st_size
    assert file_identity(tmp_path/'absent') is None and read_json(tmp_path/'absent') is None


def test_invalid_json_is_not_an_empty_state(tmp_path):
    path=tmp_path/'invalid.json'; path.write_text('[]')
    with pytest.raises(ValueError): read_json(path)


def test_alert_history_is_paged_and_filtered_without_inventing_records(tmp_path):
    path=tmp_path/'alerts.jsonl'
    records=[{'available_at':f'2020-01-01T00:0{i}:00','anomaly_score':.7+i/100} for i in range(5)]
    path.write_text('\n'.join(json.dumps(record) for record in records))
    assert read_alerts(path,offset=2,limit=2)==records[2:4]
    assert read_alerts(path,start='2020-01-01T00:03:00',end='2020-01-01T00:04:00')==records[3:]
    assert read_alerts(path,offset=10)==[]
    with pytest.raises(ValueError): read_alerts(path,limit=300)


def test_score_reduction_preserves_maxima_and_alert_counts(tmp_path):
    path=tmp_path/'scores.csv'
    data=pd.DataFrame({'timestamp':pd.date_range('2020-01-01',periods=100,freq='h'),
                       'score_max':[i/100 for i in range(100)],'scored_windows':[60]*100,'alert_windows':[i%3 for i in range(100)]})
    data.to_csv(path,index=False)
    reduced=read_score_view(path,max_points=10)
    assert len(reduced)==10 and reduced.score_max.max()==.99
    assert reduced.alert_windows.sum()==data.alert_windows.sum()
    assert reduced.scored_windows.sum()==6000
    selected=read_score_view(path,start=data.timestamp.iloc[90],end=data.timestamp.iloc[95])
    assert len(selected)==6 and selected.attrs['aggregation']=='maxima horaires'
    figure=anomaly_figure(reduced,.65)
    assert len(figure.data)==2 and figure.layout.shapes[0].y0==.65


def test_sensor_graph_breaks_gaps_without_modifying_raw_data():
    frame=pd.DataFrame({'timestamp':pd.to_datetime(['2020-01-01','2020-01-01 00:00:10','2020-01-01 00:02:00'],format='mixed'),'TP2':[8.,8.1,8.2]})
    original=frame.copy()
    figure=sensor_figure(frame,['TP2'],unit='bar')
    assert list(figure.data[0].y)==[8.,8.1,None,8.2]
    pd.testing.assert_frame_equal(frame,original)
    assert len(mode_figure({'charge probable':3}).data)==1


def test_workflow_display_marks_the_actual_human_interruption():
    state={'journal':[{'node':'monitor_node','status':'scored'},{'node':'human_review_node','status':'awaiting_review'}], 'interrupts':[{'id':'test'}]}
    figure=workflow_figure(state)
    assert len(figure.data[0].x)==7
    assert figure.data[0].customdata[5]=='En attente de validation'
    assert figure.data[0].customdata[-1]=='Non parcourue'
