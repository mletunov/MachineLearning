import React, { Component } from 'react';
import { Player } from 'video-react';
import Dropzone from 'react-dropzone';
import request from 'superagent';

class IndexComponent extends Component {

  constructor(props) {
    super(props);
    this.state = {timeStamps:[], videoSource:""};
  }

  openUploadClick(){
    this.refs.dropUploader.open();
  }

  onDrop(files){
        var req = request.post('/upload');
        files.forEach((file)=> {
            req.attach('videoFile', file);
        });
        req.end(this.fileUploaded.bind(this));
    }

  fileUploaded(err, result){

    this.setState({videoSource: result.body.filePath})
    this.getTimeStamps(result.body.fileName)
    this.refs.player.load();
  }


  getTimeStamps(fileName){
    fetch(`/time/${fileName}`)
      .then((response) =>{
          if (response.status !== 200) {
            console.log('Looks like there was a problem. Status Code: ' +  response.status);
            return;
          }

          response.json().then((stamps) =>
            {
              this.setState({timeStamps: stamps});
            }
          );
        }
      )
      .catch(function(err) {
        console.log('Fetch Error :-S', err);
      });
  }

  processNewTimeStamps(stamps){

    this.setState({timeStamps: stamps});
  }

  goToStemp(seconds) {
      this.refs.player.seek(seconds);
  }

  render() {


    let timeButtons = (this.state && this.state.timeStamps) ?
                          this.state.timeStamps.map((stamp, index) =>
                            <button key={index} type="button" className={'btn time-stamp-button ' + (stamp.fightStart == true ? 'btn-danger' : 'btn-default btn-end')} onClick={() => this.goToStemp(stamp.timeStamp)}>{stamp.fightStart == true ? 'Fight': 'Fight end'}</button>
                          ) : null

    return (
      <div>
        <nav className="navbar navbar-default">
          <div className="container">
            <div className="navbar-header">
              <a className="navbar-brand">
                Hockey Fights
              </a>
            </div>
          </div>
        </nav>
        <div className="container content-block">
            <div className="col-md-3 uploader-block">
              <Dropzone ref="dropUploader" onDrop={this.onDrop.bind(this)}>
                  <div className="drop-text-block">
                    <div className="drop-text">Drop videos here</div>
                  </div>
              </Dropzone>
              <button type="button" className="btn btn-primary upload-button" onClick={() => this.openUploadClick()}>
                Upload Video
              </button>
            </div>
            <div className="col-md-8">
               <div>
                  <Player
                      playsInline
                      ref="player"
                      src={this.state.videoSource}
                    />
                </div>
                <div className="time-stamps-row">
                    <div className="btn-group" role="group">
                      {timeButtons}
                    </div>
                </div>
            </div>
        </div>
      </div>
    );
  }
}


export default IndexComponent;
