-- Keep ::: {.only-docx} blocks for Word output and ::: {.only-latex} blocks for LaTeX output.
function Div(el)
  if el.classes:includes("only-docx") then
    if FORMAT:match("docx") then return el.content else return {} end
  end
  if el.classes:includes("only-latex") then
    if FORMAT:match("latex") then return el.content else return {} end
  end
end
