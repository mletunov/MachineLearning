import React, { Component } from 'react';

const TimeStamps = ({stamps, onTimeSeek}) => (
    <div className="time-stamps-row">
        <div className="btn-group" role="group">
            {stamps ?
                stamps.map((stamp, index) =>
                    <button key={index} type="button" className={'btn time-stamp-button ' + (stamp.fightStart == true ? 'btn-danger' : 'btn-default btn-end')} onClick={() => onTimeSeek(stamp.timeStamp)}>{stamp.fightStart == true ? 'Fight' : 'Fight end'}</button>
                ) : null}
        </div>
    </div>
);

export default TimeStamps;