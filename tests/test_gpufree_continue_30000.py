"""延期任务门禁测试：不触碰正在训练的进程或GPU。"""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from gpufree_continue_30000 import previous_ready, remaining_iterations, failed_previous

def test_target_is_total_not_additional():
    assert remaining_iterations(10000,30000)==20000
    with pytest.raises(ValueError):remaining_iterations(10000,10000)

def test_waits_for_successful_completion_and_all_old_processes_exit():
    state={"phase":"completed"}
    result={"exit_code":0,"iteration":10000,"timed_out":False}
    assert previous_ready(state,result,False)
    assert not previous_ready(state,result,True)
    assert not previous_ready({"phase":"long_running"},result,False)
    assert not previous_ready(state,{**result,"iteration":9999},False)

def test_failure_or_timeout_never_triggers_automatic_continuation():
    state={"phase":"completed"}
    result={"exit_code":0,"iteration":10000}
    assert not previous_ready(state,{**result,"exit_code":1},False)
    assert not previous_ready(state,{**result,"timed_out":True},False)
    assert failed_previous({"phase":"failed"},{})
    assert failed_previous({},{"exit_code":1})
    assert not failed_previous({"phase":"long_running"},{})
