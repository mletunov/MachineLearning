import React, { Component } from 'react';
import Dropzone from 'react-dropzone';


class Uploader extends Component {

    constructor(props) {
        super(props);
    }

    onDrop(files){
      this.props.onFileSelected(files);
    }
    onFileSelect(){
      this.refs.dropUploader.open();
    }

    render() {
        return (
            <div>
                <Dropzone ref="dropUploader" onDrop={this.onDrop.bind(this)}>
                    <div className="drop-text-block">
                        <div className="drop-text">Drop videos here</div>
                    </div>
                </Dropzone>
                <button  onClick={this.onFileSelect.bind(this)} type="button" className="btn btn-primary upload-button" >
                    Upload Video
                </button>
            </div>
        );
    }
}

export default Uploader;
