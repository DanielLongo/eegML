import {div, empty} from "core/dom"
import * as p from "core/properties"
import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"

//import {jQuery} from "jquery.min.js"
// import * as $ from "jquery";
var url = "https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js";


// create custom Bokeh model based upon example
export class KeyModResponderView extends LayoutDOMView {
    model: KeyModResponder
    
    initialize(options) {
	super.initialize(options)

	this.render()

	// Set listener so that when the a change happens
	// event, we can process the new data
	// this.connect(this.model.slider.change, () => this.render())
	// if DEBUG
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
      this.el.appendChild(div({style: 'kbd'}, 'KeyModResponder key: ' + this.model.key))
  }
}

export class KeyModResponder extends LayoutDOM {

  // If there is an associated view, this is boilerplate.
  default_view = KeyModResponderView

  // The ``type`` class attribute should generally match exactly the name
  // of the corresponding Python class.
    type = "KeyModResponder"
    
    constructor(attrs?) {
	super(attrs)
	if (attrs.parent) {
	    this.parent = attrs.parent
	} else {
	    this.parent = document
	}
    }

    initialize(options) {
	super.initialize(options)
	

	// set up to listen for keydown events, using jQuery to start but
	// may want to switch to builtin javascript 
	// bodies = document.getElementByTagName('body')
	// body = bodies[0]  // OR docuent.addEventListener
	// body.addEventListener('keydown', (ev) => this.keydown(ev))
	// need to use => function to get right this refence (or use .bind?)
	// this works
	// jQuery('body').keydown( // same as .on('keydown', handler);
	// (ev) =>  this.keydown(ev))
	// document.addEventListener('keydown', (ev) =>  this.keydown(ev)) // this works
	this.parent.addEventListener('keydown', (ev) =>  this.keydown(ev))
	// possible events keydown, keypress, keyup
	// make it easier to find this
	document.keyboardsingleton = this // this is to help me debug from console -clm
     
    }

    /* 
       keyCode: 8 key: "Backspace"
       keycallback_print:  key_num_presses 7 8 keyCode: 17 key:"Control" ctrl/shift/alt: True False False
       keycallback_print:  key_num_presses 8 9 keyCode: 16 key:"Shift" ctrl/shift/alt: False True False
       keycallback_print:  key_num_presses 9 10 keyCode: 18 key:"Alt" ctrl/shift/alt: False False True

       keycallback_print:  keyCode: 27 key:"Escape" ctrl/shift/alt: False False False key_num_presses 4 5

       keycallback_print:  key_num_presses 16 17 keyCode: 39 key:"ArrowRight" ctrl/shift/alt: False False False
       keycallback_print:  key_num_presses 17 18 keyCode: 37 key:"ArrowLeft" ctrl/shift/alt: False False False
       keycallback_print:  key_num_presses 18 19 keyCode: 37 key:"ArrowLeft" ctrl/shift/alt: False False False
       keycallback_print:  key_num_presses 19 20 keyCode: 38 key:"ArrowUp" ctrl/shift/alt: False False False
       keycallback_print:  key_num_presses 20 21 keyCode: 40 key:"ArrowDown" ctrl/shift/alt: False False False


       keycallback_print:  key_num_presses 32 33 keyCode: 34 key:"PageDown" ctrl/shift/alt: False False False
       keycallback_print:  key_num_presses 33 34 keyCode: 34 key:"PageDown" ctrl/shift/alt: False False False
       keycallback_print:  key_num_presses 34 35 keyCode: 33 key:"PageUp" ctrl/shift/alt: False False False
       keycallback_print:  key_num_presses 35 36 keyCode: 33 key:"PageUp" ctrl/shift/alt: False False False
       keycallback_print:  keyCode: 46 key:"Delete" ctrl/shift/alt: False False False key_num_presses 5 6

       keycallback_print:  key_num_presses 6 7 keyCode: 70 key:"f" ctrl/shift/alt: True False False
       keycallback_print:  key_num_presses 27 28 keyCode: 70 key:"F" ctrl/shift/alt: False True False // shift-f


       ctrl-space combo:
       keycallback_print:  key_num_presses 15 16 keyCode: 32 key:" " ctrl/shift/alt: True False False
       ctr-up combo:
       keycallback_print:  key_num_presses 22 23 keyCode: 38 key:"ArrowUp" ctrl/shift/alt: True False False


     */

    keydown(ev) {
	console.log('got keydown in KeyModResponder:', ev, this)
	// > 31 : try to filter events starting at space key (32), excludes mod keys, tab, enter, backspace, esc, break, caps lock
	if (ev.keyCode > 18) { //  filter out keys at ALT and below, allows ESC, blocks TAB, Enter, Delete but not backspace
	    this.keyCode = ev.keyCode
	    this.key=ev.key
	    this.altKey=ev.altKey
	    this.ctrlKey=ev.ctrlKey,
	    this.metaKey=ev.metaKey,
	    this.shiftKey=ev.shiftKey,
	    this.key_num_presses=this.key_num_presses+1  // works! work around to trigger change events
	}
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
KeyModResponder.define({
    keyCode : [ p.Int, -2 ],
    key : [p.String, ""],
    altKey : [p.Boolean, false],
    ctrlKey : [p.Boolean, false],
    metaKey : [p.Boolean, false],
    shiftKey : [p.Boolean, false],
    key_num_presses : [ p.Int, 0 ],
    keypress_callback : [ p.Any]
})


