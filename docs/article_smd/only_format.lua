-- Keep ::: {.only-docx} blocks for Word output and ::: {.only-latex} blocks for LaTeX output.
-- In LaTeX, give the abstract and keywords the template's look (italic, no indent); Word uses paragraph styles.
function Div(el)
  if el.classes:includes("only-docx") then
    if FORMAT:match("docx") then return el.content else return {} end
  end
  if el.classes:includes("only-latex") then
    if FORMAT:match("latex") then return el.content else return {} end
  end
  local cs = el.attributes["custom-style"]
  if FORMAT:match("latex") and (cs == "Rezumat" or cs == "CuvinteCheie") then
    local pre = (cs == "CuvinteCheie") and "\\vspace{12pt}" or ""
    local out = {pandoc.RawBlock("latex", pre .. "{\\itshape\\let\\emph\\textit")}
    for _, b in ipairs(el.content) do
      if b.t == "Para" then table.insert(b.content, 1, pandoc.RawInline("latex", "\\noindent ")) end
      table.insert(out, b)
    end
    table.insert(out, pandoc.RawBlock("latex", "\\par}"))
    return out
  end
end
