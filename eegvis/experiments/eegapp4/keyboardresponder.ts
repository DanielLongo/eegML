import {div, empty} from "core/dom"
import * as p from "core/properties"
import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"

//import {jQuery} from "jquery.min.js"
// import * as $ from "jquery";
var url = "https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js";

// load jQuery by hand
var script = document.createElement('script');

script.src = url;
script.async = false;
script.onreadystatechange = script.onload = function() {
    jQuery.noConflict();
    //declare var gLastKeyCode: any;
    // attach event listener to <body> in dom
    jQuery('body').keydown( // same as .on('keydown', handler);
	function(ev) {
            // console.log("got keydown on body via jQuery ", ev.keyCode);
	});

};
document.querySelector("head").appendChild(script);





// create custom Bokeh model based upon example
export class KeyboardResponderView extends LayoutDOMView {
    model: KeyboardResponder
    
    initialize(options) {
	super.initialize(options)

	console.log("checking that jQuery is loaded in KeyboardResponderView");
	console.log(jQuery);
	// this.model.keyCode = -1

	this.render()

	// Set listener so that when the a change happens
	// event, we can process the new data
	// this.connect(this.model.slider.change, () => this.render())
	this.connect(this.model.change, () => this.render())
    }

  render() {
      // Bokehjs Views (like Backbone) create <div> elements by default, accessible as
      // this.el
      // Many Bokeh views ignore this default <div>, and instead do things
      // like draw to the HTML canvas. In this case though, we change the
      // contents of the <div>, based on the current value.
      // this is mostly for debug purposes
      empty(this.el)
      this.el.appendChild(div({style: 'kbd'}, 'KeyboardResponder key: ' + this.model.keycode))
  }
}

export class KeyboardResponder extends LayoutDOM {

  // If there is an associated view, this is boilerplate.
  default_view = KeyboardResponderView

  // The ``type`` class attribute should generally match exactly the name
  // of the corresponding Python class.
    type = "KeyboardResponder"
    
    initialize(options) {
	super.initialize(options)
	
	console.log("checking that jQuery is loaded in KeyboardResponder");
	console.log(jQuery('body'))

	// set up to listen for keydown events, using jQuery to start but
	// may want to switch to builtin javascript 
	// bodies = document.getElementByTagName('body')
	// body = bodies[0]  // OR docuent.addEventListener
	// body.addEventListener('keydown', (ev) => this.keydown(ev))
	// need to use => function to get right this refence (or use .bind?)
	// this works
	// jQuery('body').keydown( // same as .on('keydown', handler);
	// (ev) =>  this.keydown(ev))
	document.addEventListener('keydown',
				  (ev) =>  this.keydown(ev))
	// possible events keydown, keypress, keyup
	// make it easier to find this
	document.keyboardsingleton = this
     
    }

    keydown(ev) {
	console.log('got keydown in KeyboardResponder:', ev, this)
	this.keycode = ev.keyCode
	this.key_num_presses += 1  // works! work around to trigger change events
	
	// stuff that does not work
	// this.change.emit(void 0) // based upon analogy to phosphorjs
	// try old api
	// console.log('this.change.emit:', this.change.emit) // this does not work either

	// this.keypress_callback() // this does not work it is an object not a func
	// this.keypress_callback.func() 
	// this.change.emit() // this does not work, why?

	// try this
	// this.change.emit(this.keycode)
	// console.log('emit change')
	// this.trigger_event(Event('change:keycode'))

    }

}

// The @define block adds corresponding "properties" to the JS model. These
// should basically line up 1-1 with the Python model class. Most property
// types have counterparts, e.g. bokeh.core.properties.String will be
// p.String in the JS implementation. Where the JS type system is not yet
// as rich, you can use p.Any as a "wildcard" property type.
KeyboardResponder.define({
    // text:   [ p.String ],
    // slider: [ p.Any    ],
    keycode : [ p.Int, -2 ],
    key_num_presses : [ p.Int, 0 ],
    keypress_callback : [ p.Any]
})


