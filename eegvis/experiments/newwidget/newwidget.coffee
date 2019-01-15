import * as p from "core/properties"
import {LayoutDOM, LayoutDOMVView} from "models/layouts/layout_dom"


export class NewWidget extends LayoutDOM 
  type: "NewWidget"

  @define {
    keycode: [p.Int ]
  }
