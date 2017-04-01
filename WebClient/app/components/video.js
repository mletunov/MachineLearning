import React, { Component } from 'react';
import { Player } from 'video-react';


class Video extends Component {

    constructor(props) {
        super(props);
    }

    componentWillReceiveProps(nextProps) {
        if(this.props.timeSeek !== nextProps.timeSeek)
        {
            this.player.seek(nextProps.timeSeek);
        }
    }

    render() {
        return (
            <div>
                <Player
                    playsInline
                    ref={(player) => { this.player = player; }}
                    src={this.props.filePath}
                />
            </div>
        );
    }
}

export default Video;
